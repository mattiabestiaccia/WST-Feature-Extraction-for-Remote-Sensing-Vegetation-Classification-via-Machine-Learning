#!/usr/bin/env python3
"""
Quick verification script to check generated visualizations
"""

import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'

def verify_output():
    """Verify all expected files are generated"""

    print("="*80)
    print("FEATURE VISUALIZATION OUTPUT VERIFICATION")
    print("="*80)

    # Expected patterns
    patterns = ['gradient_horizontal', 'gradient_vertical', 'checkerboard',
                'circles', 'texture', 'vertical_texture', 'edge']

    # Expected files per pattern
    expected_files = ['_original.png', '_advanced_stats.png', '_wst.png', '_comparison.png']

    total_expected = len(patterns) * len(expected_files) + 1  # +1 for overall_comparison.png
    total_found = 0
    missing_files = []

    # Check pattern directories
    for pattern in patterns:
        pattern_dir = OUTPUT_DIR / pattern

        if not pattern_dir.exists():
            print(f"\n❌ Missing directory: {pattern}")
            missing_files.extend([f"{pattern}/{pattern}{f}" for f in expected_files])
            continue

        print(f"\n✓ {pattern}/")

        for file_suffix in expected_files:
            file_path = pattern_dir / f"{pattern}{file_suffix}"

            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {file_suffix:<25} ({size_mb:.2f} MB)")
                total_found += 1
            else:
                print(f"  ❌ {file_suffix:<25} MISSING")
                missing_files.append(str(file_path.relative_to(OUTPUT_DIR)))

    # Check overall comparison
    overall_path = OUTPUT_DIR / 'overall_comparison.png'
    print(f"\n✓ Overall comparison")
    if overall_path.exists():
        size_mb = overall_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ overall_comparison.png ({size_mb:.2f} MB)")
        total_found += 1
    else:
        print(f"  ❌ overall_comparison.png MISSING")
        missing_files.append('overall_comparison.png')

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Expected files: {total_expected}")
    print(f"Found files: {total_found}")
    print(f"Missing files: {len(missing_files)}")

    if missing_files:
        print("\nMissing:")
        for f in missing_files:
            print(f"  - {f}")

    # Calculate total size
    total_size = 0
    for png_file in OUTPUT_DIR.rglob('*.png'):
        total_size += png_file.stat().st_size

    total_size_mb = total_size / (1024 * 1024)
    print(f"\nTotal disk usage: {total_size_mb:.2f} MB")

    # Success check
    if total_found == total_expected:
        print("\n✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        return True
    else:
        print(f"\n⚠️  {total_expected - total_found} files missing")
        return False

if __name__ == "__main__":
    verify_output()
