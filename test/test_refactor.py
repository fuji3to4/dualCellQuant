#!/usr/bin/env python3
"""
Test script for dualcellquant package refactoring.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import dualcellquant
        print(f"✓ dualcellquant package imported")
        print(f"  Version: {dualcellquant.__version__}")
        print(f"  Exports: {len(dualcellquant.__all__)} functions")
    except Exception as e:
        print(f"✗ Failed to import dualcellquant: {e}")
        return False
    
    # Test individual modules
    modules = ['core', 'radial', 'visualization', 'ui']
    for mod in modules:
        try:
            exec(f"from dualcellquant import {mod}")
            print(f"✓ dualcellquant.{mod} imported")
        except Exception as e:
            print(f"✗ Failed to import dualcellquant.{mod}: {e}")
            return False
    
    # Test backward compatibility
    try:
        from dualCellQuant import run_segmentation, radial_profile_analysis
        print(f"✓ Backward compatibility: dualCellQuant imports work")
    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
        return False
    
    return True

def test_core_functions():
    """Test that core functions are accessible."""
    print("\n" + "=" * 60)
    print("Testing core functions...")
    print("=" * 60)
    
    from dualcellquant import (
        get_model,
        pil_to_numpy,
        run_segmentation,
        apply_mask,
        integrate_and_quantify,
    )
    
    functions = [
        'get_model',
        'pil_to_numpy',
        'run_segmentation',
        'apply_mask',
        'integrate_and_quantify',
    ]
    
    for func_name in functions:
        func = locals()[func_name]
        if callable(func):
            print(f"✓ {func_name} is callable")
        else:
            print(f"✗ {func_name} is not callable")
            return False
    
    return True

def test_radial_functions():
    """Test that radial functions are accessible."""
    print("\n" + "=" * 60)
    print("Testing radial functions...")
    print("=" * 60)
    
    from dualcellquant import (
        radial_mask,
        radial_profile_analysis,
        radial_profile_single,
        radial_profile_all_cells,
        compute_radial_peak_difference,
    )
    
    functions = [
        'radial_mask',
        'radial_profile_analysis',
        'radial_profile_single',
        'radial_profile_all_cells',
        'compute_radial_peak_difference',
    ]
    
    for func_name in functions:
        func = locals()[func_name]
        if callable(func):
            print(f"✓ {func_name} is callable")
        else:
            print(f"✗ {func_name} is not callable")
            return False
    
    return True

def test_visualization_functions():
    """Test that visualization functions are accessible."""
    print("\n" + "=" * 60)
    print("Testing visualization functions...")
    print("=" * 60)
    
    from dualcellquant import (
        colorize_overlay,
        vivid_label_image,
        annotate_ids,
        plot_radial_profile_with_peaks,
    )
    
    functions = [
        'colorize_overlay',
        'vivid_label_image',
        'annotate_ids',
        'plot_radial_profile_with_peaks',
    ]
    
    for func_name in functions:
        func = locals()[func_name]
        if callable(func):
            print(f"✓ {func_name} is callable")
        else:
            print(f"✗ {func_name} is not callable")
            return False
    
    return True

def test_ui():
    """Test that UI can be built."""
    print("\n" + "=" * 60)
    print("Testing UI...")
    print("=" * 60)
    
    try:
        from dualcellquant import build_ui
        print(f"✓ build_ui imported")
        
        demo = build_ui()
        print(f"✓ UI built successfully")
        print(f"  Type: {type(demo).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to build UI: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DualCellQuant Package Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Core Functions", test_core_functions),
        ("Radial Functions", test_radial_functions),
        ("Visualization Functions", test_visualization_functions),
        ("UI", test_ui),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
