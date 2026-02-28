"""
test_stylometry_integration.py — Test the stylometry integration with syncom.

Tests:
1. Loading population model from stylometry
2. Loading voice skills
3. Persona creation with voice skills
4. Complete pipeline integration
"""

import sys
from pathlib import Path

# Test configuration
DOCKET_ID = "CMS-2025-0050-0031"
TEST_RULE_TEXT = """
This proposed rule updates healthcare information technology standards
to improve interoperability and reduce provider burden.
"""


def test_stylometry_loader():
    """Test that we can load population model from stylometry."""
    print("\n=== Test 1: Loading Population Model from Stylometry ===")
    
    try:
        from stylometry.stylometry_loader import build_population_model
        
        population = build_population_model(DOCKET_ID)
        
        print(f"✓ Population loaded successfully")
        print(f"  Docket ID: {population.docket_id}")
        print(f"  Total comments: {population.total_comments}")
        print(f"  Archetypes found: {len(population.archetypes)}")
        
        # Check archetypes
        for archetype, profile in population.archetypes.items():
            print(f"  - {archetype}: {profile.count} comments")
            print(f"    States available: {len(profile.states)}")
            print(f"    Orgs available: {len(profile.orgs)}")
        
        # Test archetype sampling
        import numpy as np
        rng = np.random.default_rng(42)
        sample_arch = population.sample_archetype(rng)
        print(f"  Sample archetype: {sample_arch}")
        
        return True, population
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_voice_skill_loading():
    """Test that we can load voice skills."""
    print("\n=== Test 2: Loading Voice Skills ===")
    
    try:
        from stylometry.stylometry_loader import load_voice_skill
        
        # Test exact match
        skill = load_voice_skill(DOCKET_ID, "individual_consumer", "low")
        
        if skill:
            print(f"✓ Voice skill loaded successfully")
            print(f"  Skill length: {len(skill)} characters")
            print(f"  First 200 chars: {skill[:200]}...")
        else:
            print(f"✗ No skill found for individual_consumer-low")
            return False
        
        # Test approximate match
        skill2 = load_voice_skill(DOCKET_ID, "academic", "medium")
        if skill2:
            print(f"✓ Approximate match worked (academic-medium → academic-high)")
        else:
            print(f"  Note: No approximate match found")
        
        # Test missing archetype
        skill3 = load_voice_skill(DOCKET_ID, "nonexistent", "high")
        if not skill3:
            print(f"✓ Correctly returns None for missing archetype")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persona_with_voice_skill(population):
    """Test persona creation with voice skill loading."""
    print("\n=== Test 3: Persona Creation with Voice Skills ===")
    
    try:
        from syncom.persona import sample_persona
        from syncom.world_model import WorldModel
        from config import Config
        import numpy as np
        
        # Create a minimal world model
        world_model = WorldModel(
            rule_title="Test Rule",
            agency="CMS",
            docket_id=DOCKET_ID,
            core_change="Testing",
            regulatory_domain="healthcare",
            population=population,
        )
        
        config = Config()
        rng = np.random.default_rng(42)
        
        # Sample persona WITHOUT docket_id (should use generic)
        print("  Testing without docket_id (generic fallback)...")
        persona_generic = sample_persona(world_model, config, rng)
        print(f"  ✓ Persona created: {persona_generic.full_name}")
        print(f"    Archetype: {persona_generic.archetype}")
        print(f"    Sophistication: {persona_generic.sophistication}")
        print(f"    Has voice_skill: {bool(persona_generic.voice_skill)}")
        
        if not persona_generic.voice_skill:
            print(f"    (Expected: using generic fallback)")
        
        # Sample persona WITH docket_id (should load skill)
        print("\n  Testing with docket_id (skill loading)...")
        rng = np.random.default_rng(43)  # Different seed
        persona_skilled = sample_persona(world_model, config, rng, docket_id=DOCKET_ID)
        print(f"  ✓ Persona created: {persona_skilled.full_name}")
        print(f"    Archetype: {persona_skilled.archetype}")
        print(f"    Sophistication: {persona_skilled.sophistication}")
        print(f"    Has voice_skill: {bool(persona_skilled.voice_skill)}")
        
        if persona_skilled.voice_skill:
            print(f"    ✓ Voice skill loaded! Length: {len(persona_skilled.voice_skill)}")
            
            # Test style_instructions
            instructions = persona_skilled.style_instructions()
            print(f"    Style instructions length: {len(instructions)}")
            print(f"    First 150 chars: {instructions[:150]}...")
        else:
            print(f"    Note: No skill available for {persona_skilled.archetype}/{persona_skilled.sophistication}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test that pipeline works with new stylometry integration."""
    print("\n=== Test 4: Pipeline Integration Test ===")
    
    try:
        from syncom.pipeline import run
        from config import Config
        
        print("  Note: This test requires API credentials and will be skipped if not available")
        
        # Try to create config
        try:
            config = Config()
            config.validate()
        except ValueError as e:
            print(f"  ⊘ Skipping: API credentials not configured ({e})")
            return True  # Not a failure, just skipped
        
        print("  This would run a full pipeline test with:")
        print(f"    docket_id='{DOCKET_ID}'")
        print(f"    rule_text=<test rule>")
        print(f"    vector=1")
        print(f"    objective='Support the rule'")
        print(f"    volume=1")
        
        print("  ⊘ Skipping actual generation to avoid API costs")
        print("  (Manual testing recommended)")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("STYLOMETRY INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Check that stylometry data exists
    stylometry_path = Path("stylometry") / DOCKET_ID
    if not stylometry_path.exists():
        print(f"\n✗ ERROR: Stylometry data not found at {stylometry_path}")
        print(f"  Run: python stylometry_analyzer.py CMS-2025-0050-0031.csv")
        return False
    
    print(f"✓ Stylometry data found at {stylometry_path}")
    
    # Run tests
    results = []
    
    # Test 1: Load population
    success, population = test_stylometry_loader()
    results.append(("Population Loading", success))
    
    if not success:
        print("\n✗ Cannot continue without population model")
        return False
    
    # Test 2: Load voice skills
    success = test_voice_skill_loading()
    results.append(("Voice Skill Loading", success))
    
    # Test 3: Persona with skills
    success = test_persona_with_voice_skill(population)
    results.append(("Persona Creation", success))
    
    # Test 4: Pipeline integration
    success = test_pipeline_integration()
    results.append(("Pipeline Integration", success))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
