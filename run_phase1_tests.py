#!/usr/bin/env python3
"""
Phase 1 Test Runner
Runs all Phase 1 trading strategy tests
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all Phase 1 tests"""

    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    print("🚀 Running Phase 1 Trading Strategy Tests")
    print("=" * 50)

    # Test files to run
    test_files = [
        "test_phase1_strategies.py",
        "test_phase1_integration.py",
        "test_system.py"  # Include existing system tests
    ]

    results = []

    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue

        print(f"\n📋 Running {test_file}...")
        print("-" * 30)

        try:
            # Run pytest on the test file
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short"
            ], capture_output=True, text=True, cwd=project_root)

            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
                results.append(True)
            else:
                print(f"❌ {test_file} FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results.append(False)

        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_file, result) in enumerate(zip(test_files, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print("2")

    print(f"\n🎯 Overall Result: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ Ready to proceed to Phase 2")
        return 0
    else:
        print("⚠️  Some tests failed. Please review and fix before proceeding.")
        return 1

def run_specific_test(test_name):
    """Run a specific test"""
    project_root = Path(__file__).parent.parent

    print(f"🎯 Running specific test: {test_name}")

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"tests/{test_name}",
            "-v",
            "--tb=short"
        ], cwd=project_root)

        return result.returncode

    except Exception as e:
        print(f"Error running test {test_name}: {e}")
        return 1

def install_dependencies():
    """Install test dependencies if needed"""
    print("📦 Installing test dependencies...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)

        print("✅ Dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install":
            install_dependencies()
        elif sys.argv[1] == "--specific":
            if len(sys.argv) > 2:
                sys.exit(run_specific_test(sys.argv[2]))
            else:
                print("Usage: python run_phase1_tests.py --specific <test_file>")
                sys.exit(1)
        else:
            print("Usage: python run_phase1_tests.py [--install] [--specific <test_file>]")
            sys.exit(1)
    else:
        # Install dependencies first
        if not install_dependencies():
            sys.exit(1)

        # Run all tests
        sys.exit(run_tests())
