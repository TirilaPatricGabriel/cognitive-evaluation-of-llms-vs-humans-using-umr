from app.services.umr_parser import create_umr_parser
from app.utils.umr_visualizer import UMRVisualizer

def test_umr_parsing():

    parser = create_umr_parser()
    visualizer = UMRVisualizer()

    # Test English
    print("=" * 80)
    print("TESTING ENGLISH UMR PARSING")
    print("=" * 80)

    en_text = "After he finished eating, he left immediately."
    print(f"\nInput text: {en_text}")
    print("\nParsing...")

    result_en = parser.parse_text(en_text, "english")

    print(f"\nSuccess: {result_en.get('success', False)}")
    print(f"\nUMR Graph:\n{result_en.get('umr_graph', 'N/A')}")

    if result_en.get('umr_graph'):
        structure = visualizer.parse_penman_structure(result_en['umr_graph'])
        print(f"\nStructural Analysis:")
        print(f"  - Depth: {structure['stats']['depth']}")
        print(f"  - Concepts: {structure['stats']['num_concepts']}")
        print(f"  - Roles: {structure['stats']['num_roles']}")
        print(f"  - Has Aspect: {structure['stats']['has_aspect']}")
        print(f"  - Has Temporal: {structure['stats']['has_temporal']}")
        print(f"  - Main Concept: {structure['main_concept']}")

    # Test Romanian
    print("\n" + "=" * 80)
    print("TESTING ROMANIAN UMR PARSING")
    print("=" * 80)

    ro_text = "După ce a terminat de mâncat, a plecat imediat."
    print(f"\nInput text: {ro_text}")
    print("\nParsing...")

    result_ro = parser.parse_text(ro_text, "romanian")

    print(f"\nSuccess: {result_ro.get('success', False)}")
    print(f"\nUMR Graph:\n{result_ro.get('umr_graph', 'N/A')}")

    if result_ro.get('umr_graph'):
        structure = visualizer.parse_penman_structure(result_ro['umr_graph'])
        print(f"\nStructural Analysis:")
        print(f"  - Depth: {structure['stats']['depth']}")
        print(f"  - Concepts: {structure['stats']['num_concepts']}")
        print(f"  - Roles: {structure['stats']['num_roles']}")
        print(f"  - Has Aspect: {structure['stats']['has_aspect']}")
        print(f"  - Has Temporal: {structure['stats']['has_temporal']}")
        print(f"  - Main Concept: {structure['main_concept']}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_umr_parsing()
