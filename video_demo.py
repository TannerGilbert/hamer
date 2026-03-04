import argparse
import os
from pathlib import Path

import cv2
import ffmpeg
import numpy as np
import torch
from tqdm import tqdm

torch.cuda.empty_cache()
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel


def get_fps_accurate(video_path):
    # from https://gist.github.com/realphongha/f8db807a9a0fc512ed146786e684cd43
    # using opencv-python and ffmpeg-python packages, slow but more accurate
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open {video_path}")
    metadata = ffmpeg.probe(video_path)
    duration = metadata["format"]["duration"]
    duration = float(duration)
    if duration == 0:
        raise ZeroDivisionError("Video duration is zero!")
    frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frames += 1
    cap.release()
    return frames / duration, frames


def process_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    model,
    model_cfg,
    detector,
    cpm,
    renderer,
    device,
    args,
) -> np.ndarray:
    det_out = detector(frame_bgr)
    frame_rgb = frame_bgr[:, :, ::-1]

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if pred_bboxes.shape[0] == 0:
        return frame_bgr

    vitposes_out = cpm.predict_pose(
        frame_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if np.sum(valid) > 3:
            bboxes.append(
                [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            )
            is_right.append(0)

        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if np.sum(valid) > 3:
            bboxes.append(
                [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            )
            is_right.append(1)

    if len(bboxes) == 0:
        return frame_bgr

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, right, rescale_factor=args.rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    scaled_focal_length = None
    render_res = None

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length,
        ).detach().cpu().numpy()
        render_res = img_size[0]

        batch_size = batch["img"].shape[0]
        for n in range(batch_size):
            verts = out["pred_vertices"][n].detach().cpu().numpy()
            hand_side = batch["right"][n].cpu().numpy()
            verts[:, 0] = (2 * hand_side - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(hand_side)

            if args.save_mesh:
                tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE, is_right=hand_side)
                tmesh.export(os.path.join(args.out_folder, f"frame_{frame_idx:06d}_hand_{int(batch['personid'][n]):02d}.obj"))

    if len(all_verts) == 0 or scaled_focal_length is None or render_res is None:
        return frame_bgr

    cam_view = renderer.render_rgba_multiple(
        all_verts,
        cam_t=all_cam_t,
        render_res=render_res,
        is_right=all_right,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )

    input_img = frame_bgr.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    overlay_rgb = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    overlay_bgr = (255.0 * overlay_rgb[:, :, ::-1]).clip(0, 255).astype(np.uint8)
    return overlay_bgr


def main():
    parser = argparse.ArgumentParser(description="HaMeR video demo code")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to pretrained model checkpoint")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--out_folder", type=str, default="out_demo", help="Output folder")
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument("--out_video_name", type=str, default="output_overlay.mp4", help="Output video filename")
    parser.add_argument("--save_mesh", dest="save_mesh", action="store_true", default=False, help="If set, save meshes to disk")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference/fitting")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox")
    parser.add_argument(
        "--body_detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime and reduces memory",
    )

    args = parser.parse_args()

    model, model_cfg = load_hamer(args.checkpoint)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == "vitdet":
        import hamer
        from detectron2.config import LazyConfig

        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg, device)
    else:
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg, device)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    os.makedirs(args.out_folder, exist_ok=True)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.video_path}")

    fps, total_frames = get_fps_accurate(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(args.out_folder, args.out_video_name)
    
    print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
    
    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video: {out_video_path}")

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out_frame = process_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                model=model,
                model_cfg=model_cfg,
                detector=detector,
                cpm=cpm,
                renderer=renderer,
                device=device,
                args=args,
            )
            writer.write(out_frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
    print(f"Saved overlay video to: {out_video_path}")
    print(f"Processed frames: {frame_idx}")


if __name__ == "__main__":
    main()
