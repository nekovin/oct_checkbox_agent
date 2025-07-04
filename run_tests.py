#!/usr/bin/env python3
"""Test runner script for OCR Checkbox Agent."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all tests using pytest."""
    
    # Change to the ocr_checkbox_agent directory
    ocr_dir = Path(__file__).parent / "ocr_checkbox_agent"
    
    if not ocr_dir.exists():
        print("❌ OCR Checkbox Agent directory not found!")
        return False
    
    print("🧪 Running OCR Checkbox Agent Tests")
    print("=" * 50)
    
    # Run pytest with various options
    cmd = [
        sys.executable, "-m", "pytest",
        str(ocr_dir / "tests"),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker handling
        "-x",  # Stop on first failure
    ]
    
    try:
        result = subprocess.run(cmd, cwd=ocr_dir, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "pytest",
        "numpy", 
        "opencv-python",
        "pandas",
        "pydantic",
        "loguru"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r ocr_checkbox_agent/requirements.txt")
        return False
    
    print("✅ All required packages are available")
    return True


def main():
    """Main test runner."""
    
    print("OCR Checkbox Agent - Test Runner")
    print("=" * 40)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot run tests due to missing dependencies")
        sys.exit(1)
    
    print()
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n🎉 Test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()