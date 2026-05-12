"""
Example Usage: Unified Damage Detection Pipeline

This script demonstrates all the key features of the detection pipeline.
Run with: python examples/pipeline_example.py
"""

import sys
import json
import time
from PIL import Image
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.detection_pipeline import (
    get_pipeline,
    DamageDetectionPipeline,
)


def example_1_basic_usage():
    """Example 1: Basic pipeline usage."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*70)
    
    # Initialize pipeline
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
        confidence_threshold=0.25,
        iou_threshold=0.1,
    )
    
    # Load sample image (replace with actual image path)
    image_path = "test_image.jpg"  # You'll need to provide this
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Create a test image or update the path")
        return
    
    # Process image
    print(f"Processing: {image_path}")
    image = Image.open(image_path).convert("RGB")
    result = pipeline.process(image, imgsz=640)
    
    # Display results
    print(f"\n✅ Results:")
    print(f"   Damages detected: {len(result.damage_detections)}")
    print(f"   Parts detected: {len(result.part_detections)}")
    print(f"   Associations found: {len(result.associations)}")
    print(f"   Processing time: {result.processing_time_ms:.2f}ms")
    
    return result


def example_2_detailed_associations():
    """Example 2: Detailed analysis of associations."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Detailed Association Analysis")
    print("="*70)
    
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
    )
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    result = pipeline.process(image)
    
    print(f"\n📊 Damage-Part Associations:")
    print("-" * 70)
    
    if len(result.associations) == 0:
        print("No associations found")
        return
    
    for i, assoc in enumerate(result.associations, 1):
        print(f"\n{i}. Association:")
        print(f"   Damage: {assoc.damage_detection.class_name} "
              f"(conf: {assoc.damage_detection.confidence:.1%})")
        print(f"   Part: {assoc.part_detection.class_name} "
              f"(conf: {assoc.part_detection.confidence:.1%})")
        print(f"   IoU: {assoc.iou:.1%}")
        print(f"   Overlap: {assoc.overlap_percentage:.1f}%")
        print(f"   Damage bbox: {assoc.damage_detection.bbox.to_list()}")
        print(f"   Part bbox: {assoc.part_detection.bbox.to_list()}")


def example_3_filtering_by_severity():
    """Example 3: Filter associations by severity."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Severity-Based Filtering")
    print("="*70)
    
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
    )
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    result = pipeline.process(image)
    
    # Define damage severity
    damage_severity = {
        "tire flat": "critical",
        "glass shatter": "critical",
        "lamp broken": "high",
        "crack": "high",
        "dent": "medium",
        "scratch": "low",
    }
    
    # Group by severity
    severity_groups = {}
    for assoc in result.associations:
        damage_type = assoc.damage_detection.class_name
        severity = damage_severity.get(damage_type, "unknown")
        
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(assoc)
    
    # Display grouped results
    print("\n🔴 By Severity Level:")
    for severity_level in ["critical", "high", "medium", "low"]:
        assocs = severity_groups.get(severity_level, [])
        if assocs:
            print(f"\n{severity_level.upper()} ({len(assocs)} issues):")
            for assoc in assocs:
                print(f"  • {assoc.damage_detection.class_name} on "
                      f"{assoc.part_detection.class_name}")


def example_4_critical_parts_analysis():
    """Example 4: Focus on critical car parts."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Critical Parts Analysis")
    print("="*70)
    
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
    )
    
    # Define critical parts
    CRITICAL_PARTS = {
        "Windshield", "Front Windshield", "Rear Glass",
        "Headlight", "Taillight", "Fog Lamp",
        "Front Bumper", "Rear Bumper",
    }
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    result = pipeline.process(image)
    
    # Filter to critical parts only
    critical_assocs = [
        assoc for assoc in result.associations
        if assoc.part_detection.class_name in CRITICAL_PARTS
    ]
    
    print(f"\n🚨 Critical Parts Affected: {len(critical_assocs)}")
    
    if critical_assocs:
        for assoc in critical_assocs:
            print(f"\n⚠️  {assoc.part_detection.class_name}")
            print(f"   Damage: {assoc.damage_detection.class_name}")
            print(f"   Severity: IoU {assoc.iou:.0%}, "
                  f"Overlap {assoc.overlap_percentage:.1f}%")


def example_5_threshold_tuning():
    """Example 5: Find optimal thresholds."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Threshold Tuning")
    print("="*70)
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    
    print("\nTesting different confidence thresholds:")
    print("-" * 70)
    print(f"{'Conf':>6} | {'Damages':>7} | {'Parts':>5} | {'Assoc':>5} | Time(ms)")
    print("-" * 70)
    
    for conf_thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        pipeline = DamageDetectionPipeline(
            damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
            part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
            confidence_threshold=conf_thresh,
            iou_threshold=0.1,
        )
        
        start = time.time()
        result = pipeline.process(image, imgsz=640)
        elapsed = (time.time() - start) * 1000
        
        print(f"{conf_thresh:>6.2f} | {len(result.damage_detections):>7} | "
              f"{len(result.part_detections):>5} | "
              f"{len(result.associations):>5} | {elapsed:>7.1f}")


def example_6_batch_processing():
    """Example 6: Process multiple images."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Batch Processing")
    print("="*70)
    
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
    )
    
    # Find test images
    test_dir = Path("test_images")
    if not test_dir.exists():
        print(f"⚠️  Directory not found: {test_dir}")
        print("   Create a 'test_images' directory with .jpg files")
        return
    
    image_paths = list(test_dir.glob("*.jpg"))
    if not image_paths:
        print(f"⚠️  No .jpg files found in {test_dir}")
        return
    
    print(f"\nProcessing {len(image_paths)} images...\n")
    
    results_summary = []
    total_time = 0
    
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        result = pipeline.process(image)
        
        summary = {
            "filename": img_path.name,
            "damages": len(result.damage_detections),
            "parts": len(result.part_detections),
            "associations": len(result.associations),
            "time_ms": result.processing_time_ms,
        }
        results_summary.append(summary)
        total_time += result.processing_time_ms
        
        print(f"✓ {img_path.name:40} | "
              f"D:{summary['damages']:2} P:{summary['parts']:2} "
              f"A:{summary['associations']:2} | {summary['time_ms']:6.1f}ms")
    
    # Summary statistics
    avg_time = total_time / len(image_paths)
    avg_damages = sum(s["damages"] for s in results_summary) / len(image_paths)
    avg_assocs = sum(s["associations"] for s in results_summary) / len(image_paths)
    
    print("\n" + "-" * 70)
    print(f"Summary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total time: {total_time:.1f}ms")
    print(f"  Avg time/image: {avg_time:.1f}ms")
    print(f"  Avg damages/image: {avg_damages:.1f}")
    print(f"  Avg associations/image: {avg_assocs:.1f}")


def example_7_export_json():
    """Example 7: Export results as JSON."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Export Results as JSON")
    print("="*70)
    
    pipeline = DamageDetectionPipeline(
        damage_model_path=str(PROJECT_ROOT / "runs/damage/weights/best.pt"),
        part_model_path=str(PROJECT_ROOT / "runs/parts/weights/best.pt"),
    )
    
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    result = pipeline.process(image)
    
    # Convert to JSON
    output_dict = result.to_dict()
    
    # Save to file
    output_path = "detection_result.json"
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\n✅ Results exported to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size} bytes")
    
    # Display sample
    print(f"\nSample JSON structure:")
    print(json.dumps({
        "summary": output_dict["summary"],
        "damage_detection_example": output_dict["damage_detections"][0] if output_dict["damage_detections"] else None,
    }, indent=2))


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("UNIFIED DAMAGE DETECTION PIPELINE - EXAMPLES")
    print("="*70)
    
    print("\n💡 Available examples:")
    print("   1. Basic pipeline usage")
    print("   2. Detailed association analysis")
    print("   3. Severity-based filtering")
    print("   4. Critical parts analysis")
    print("   5. Threshold tuning")
    print("   6. Batch processing")
    print("   7. Export as JSON")
    
    choice = input("\nEnter example number (1-7) or 'all': ").strip()
    
    examples = {
        "1": example_1_basic_usage,
        "2": example_2_detailed_associations,
        "3": example_3_filtering_by_severity,
        "4": example_4_critical_parts_analysis,
        "5": example_5_threshold_tuning,
        "6": example_6_batch_processing,
        "7": example_7_export_json,
    }
    
    if choice.lower() == "all":
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
