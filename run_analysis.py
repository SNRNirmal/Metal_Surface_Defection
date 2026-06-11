
import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt

from DefectAnalysis.pipeline import ONNXInferencePipeline

ONNX_MODEL_MAP = {
    "fpn_resnet34": "ONNX_models/fpn_resnet.onnx",
}

VAL_IMAGE_DIR = "Validation_Images"
OUTPUT_DIR    = "outputs/analysis"


def parse_args():
    parser = argparse.ArgumentParser(description="Steel defect analysis demo")
    parser.add_argument("--model",     default="fpn_resnet34",
                        choices=list(ONNX_MODEL_MAP.keys()))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_px",    type=int,   default=20,
                        help="Minimum defect component size in pixels")
    parser.add_argument("--save",      action="store_true",
                        help="Save overlay + heatmap images")
    parser.add_argument("--show",      action="store_true",
                        help="Display results with matplotlib")
    return parser.parse_args()


def display_result(original, overlay, heatmap, report, rul, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13)

    for ax, img, label in zip(
        axes,
        [original, overlay, heatmap],
        ["Original", "Defect Overlay", "Heatmap"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")

    summary = (
        f"Defect: {report['defect_percentage']:.2f}%  |  "
        f"Severity: {report['severity']}  |  "
        f"Risk: {report['risk_score']:.1f}/100  |  "
        f"RUL: {rul['rul_days']} days"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    model_path = ONNX_MODEL_MAP[args.model]
    if not os.path.isfile(model_path):
        print(f"❌ ONNX model not found: {model_path}")
        print("   Export it first:  python -m Utilities.ONNX_converter ...")
        sys.exit(1)

    pipe = ONNXInferencePipeline(
        model_path,
        threshold=args.threshold,
        min_defect_pixels=args.min_px,
    )

    images = [
        f for f in os.listdir(VAL_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        print(f"❌ No images found in '{VAL_IMAGE_DIR}'")
        sys.exit(1)

    print(f"\n📂 Found {len(images)} validation images")
    print(f"🤖 Model     : {args.model}")
    print(f"⚙️  Threshold : {args.threshold}")
    print("-" * 60)

    for fname in sorted(images):
        path = os.path.join(VAL_IMAGE_DIR, fname)
        print(f"\n🔍 {fname}")

        result = pipe.run(
            path,
            save_dir=OUTPUT_DIR if args.save else None,
        )

        report = result["report"]
        rul    = result["rul"]

        # Print report
        print(f"   Defect Area   : {report['defect_percentage']:.2f}%")
        print(f"   Severity      : {report['severity']}")
        print(f"   Risk Score    : {report['risk_score']:.2f} / 100")
        print(f"   RUL (rule)    : {rul['rul_days']} days")
        print(f"   Recommendation: {rul['rul_recommendation']}")

        non_zero = {k: v for k, v in report["per_class_area"].items() if v > 0}
        if non_zero:
            print("   Class areas   :")
            for cls, pct in sorted(non_zero.items(), key=lambda x: -x[1]):
                print(f"     {cls:<20} {pct:.4f}%")
        else:
            print("   No defects detected.")

        if args.show:
            original = cv2.imread(path)
            display_result(
                original,
                result["overlay_image"],
                result["heatmap_image"],
                report, rul,
                title=fname,
            )

    print("\n✅ Done.")
    if args.save:
        print(f"💾 Results saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()