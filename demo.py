# NOTE: Requires SAM-2 and RAFT installed. See README for setup.
# pip install git+https://github.com/facebookresearch/segment-anything-2.git

"""
SAMFlow Demo â€” zero-shot video segmentation with SAM-2 + optical flow.
Usage: python demo.py --video path/to/video.mp4 --prompt click|box
"""
import argparse, cv2
import numpy as np

def demo_segment(video_path: str, prompt_type: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} FPS")
    print(f"Prompt type: {prompt_type}")
    print()
    print("Pipeline:")
    print("  1. Load SAM-2 image encoder")
    print("  2. Get user prompt on frame 0 (click or bounding box)")
    print("  3. SAM-2 decoder â†’ initial mask")
    print("  4. For each subsequent frame:")
    print("     a. RAFT optical flow: warp previous mask â†’ spatial hint")
    print("     b. SAM-2 memory attention: cross-attend to stored embeddings")
    print("     c. SAM-2 decoder â†’ propagated mask")
    print("  5. Render coloured overlay onto video")
    print()
    print("To run inference, install SAM-2:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    print("  python segment.py --video", video_path)
    cap.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAMFlow Demo")
    p.add_argument("--video",  required=True, help="Input video path")
    p.add_argument("--prompt", default="click", choices=["click","box"])
    a = p.parse_args()
    demo_segment(a.video, a.prompt)
