#!/usr/bin/env python3
"""
print_ontology.py

Simple test script to read and display ontology classes from a Turtle (.ttl) file.

Usage:
    python print_ontology.py <path_to_ontology.ttl> [--detailed]

Example:
    python print_ontology.py ontology.ttl
    python print_ontology.py ontology.ttl --detailed
"""

import sys
import argparse

from savant_common.ontology import read_ontology_classes


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read and display ontology classes from a Turtle file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python print_ontology.py ontology.ttl
  python print_ontology.py ontology.ttl --detailed
  python print_ontology.py ontology.ttl --top-level DynamicObject
        """
    )

    parser.add_argument('ttl_file',
                       help='Path to the Turtle ontology file (.ttl)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed information including parent and top-level classes')
    parser.add_argument('--top-level', '-t',
                       help='Filter to show only classes with specific top-level class')

    return parser.parse_args()


def print_classes_simple(classes):
    """Print ontology classes in simple format."""
    if not classes:
        print("No classes found in the ontology file.")
        return

    print(f"\nFound {len(classes)} ontology class(es):\n")
    print("=" * 80)

    for i, cls in enumerate(classes, 1):
        print(f"\nClass #{i}:")
        print(f"  Class Name: {cls['class_name']}")
        print(f"  Label:      {cls['label']}")
        print(f"  UID:        {cls['uid']}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(classes)} class(es)")


def print_classes_detailed(classes):
    """Print ontology classes with detailed information."""
    if not classes:
        print("No classes found in the ontology file.")
        return

    print(f"\nFound {len(classes)} ontology class(es):\n")
    print("=" * 100)

    for i, cls in enumerate(classes, 1):
        print(f"\nClass #{i}:")
        print(f"  Class Name:       {cls['class_name']}")
        print(f"  Label:            {cls['label']}")
        print(f"  UID:              {cls['uid']}")
        print(f"  Parent Class:     {cls['parent_class_name'] or 'None (top-level)'}")
        print(f"  Top-Level Class:  {cls['top_level_class_name']}")
        if cls['uid']:
            print(f"  Full URI:         {cls['uri']}")

    print("\n" + "=" * 100)
    print(f"\nTotal: {len(classes)} class(es)")


def print_hierarchy_summary(classes):
    """Print a summary of the class hierarchy."""
    # Group by top-level class
    hierarchy = {}
    for cls in classes:
        top = cls['top_level_class_name'] or 'Unknown'
        if top not in hierarchy:
            hierarchy[top] = []
        hierarchy[top].append(cls)

    print("\n" + "=" * 100)
    print("HIERARCHY SUMMARY")
    print("=" * 100)

    for top_level, members in sorted(hierarchy.items()):
        print(f"\n{top_level} ({len(members)} class(es))")
        print("-" * 80)
        for cls in sorted(members, key=lambda x: x['uid'] or 0):
            indent = "  "
            if cls['parent_class_name'] and cls['parent_class_name'] != top_level:
                indent = "    "
            print(f"{indent}{cls['class_name']:<30} (UID: {cls['uid']:<4}) Label: {cls['label']}")


def main():
    """Main function."""
    try:
        args = parse_arguments()

        print(f"Reading ontology from: {args.ttl_file}")

        # Read ontology classes
        classes = read_ontology_classes(args.ttl_file)

        # Filter by top-level class if specified
        if args.top_level:
            classes = [c for c in classes if c['top_level_class_name'] == args.top_level]
            print(f"Filtering to show only classes under top-level: {args.top_level}")

        # Print results
        if args.detailed:
            print_classes_detailed(classes)
            print_hierarchy_summary(classes)
        else:
            print_classes_simple(classes)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
