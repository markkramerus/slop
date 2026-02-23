#!/usr/bin/env python3
"""
Quick test script to verify async implementation works correctly.
This doesn't make real API calls - just validates the structure.
"""

import sys
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

def test_imports():
    """Test that all async functions can be imported."""
    print("Testing imports...")
    
    try:
        from slop.config import Config
        from slop.generator import generate_comment_async
        from slop.persona import sample_persona_async
        from slop.argument_mapper import map_argument_async
        from slop.quality_control import QualityController
        from slop.pipeline import run_async, _generate_one_comment_async
        print("✓ All async imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_async_functions():
    """Test that async functions are actually async."""
    print("\nTesting async function signatures...")
    
    from slop.generator import generate_comment_async
    from slop.persona import sample_persona_async
    from slop.argument_mapper import map_argument_async
    from slop.quality_control import QualityController
    
    # Check if functions are coroutines
    checks = []
    
    # These should be async functions (coroutine functions)
    checks.append(asyncio.iscoroutinefunction(generate_comment_async))
    checks.append(asyncio.iscoroutinefunction(sample_persona_async))
    checks.append(asyncio.iscoroutinefunction(map_argument_async))
    
    # QC should have async method
    qc = QualityController(
        config=Mock(),
        objective="test",
        skip_relevance_check=True,
        skip_argument_check=True,
        skip_embedding_check=True,
    )
    checks.append(asyncio.iscoroutinefunction(qc.check_async))
    
    if all(checks):
        print("✓ All async functions properly defined")
        return True
    else:
        print("✗ Some functions are not async")
        return False


def test_config_clients():
    """Test that config has async client methods."""
    print("\nTesting config async clients...")
    
    from slop.config import Config
    
    # Create config (will fail validation without keys, but that's ok for this test)
    config = Config()
    config.api_key = "test"
    config.embed_api_key = "test"
    
    # Check methods exist
    has_async_openai = hasattr(config, 'async_openai_client')
    has_async_embedding = hasattr(config, 'async_embedding_client')
    
    if has_async_openai and has_async_embedding:
        print("✓ Config has async client methods")
        return True
    else:
        print("✗ Config missing async client methods")
        return False


def test_cli_flags():
    """Test that CLI has new flags."""
    print("\nTesting CLI flags...")
    
    from cli import build_parser
    
    parser = build_parser()
    
    # Parse with new flags
    try:
        args = parser.parse_args([
            '--docket-csv', 'test.csv',
            '--rule-text', 'test',
            '--vector', '2',
            '--objective', 'test',
            '--volume', '10',
            '--output', 'test.csv',
            '--max-concurrent', '15',
            '--no-async',
        ])
        
        has_max_concurrent = hasattr(args, 'max_concurrent')
        has_no_async = hasattr(args, 'no_async')
        correct_values = args.max_concurrent == 15 and args.no_async == True
        
        if has_max_concurrent and has_no_async and correct_values:
            print("✓ CLI flags properly configured")
            return True
        else:
            print("✗ CLI flags not properly configured")
            return False
            
    except Exception as e:
        print(f"✗ CLI parsing failed: {e}")
        return False


def test_pipeline_has_run_async():
    """Test that pipeline has run_async function."""
    print("\nTesting pipeline has run_async...")
    
    from slop.pipeline import run_async
    
    # Check signature
    import inspect
    sig = inspect.signature(run_async)
    has_max_concurrent = 'max_concurrent' in sig.parameters
    
    if has_max_concurrent:
        print("✓ Pipeline has run_async with max_concurrent parameter")
        return True
    else:
        print("✗ Pipeline missing proper run_async implementation")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Async Implementation Validation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_async_functions,
        test_config_clients,
        test_cli_flags,
        test_pipeline_has_run_async,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        print("✓ Async implementation is properly structured")
        print("\nNext steps:")
        print("  1. Run with a small volume to test actual API integration")
        print("  2. Compare timing: --no-async vs default async mode")
        print("  3. Verify output quality is identical between modes")
        return 0
    else:
        print("\n✗ Some tests failed - review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
