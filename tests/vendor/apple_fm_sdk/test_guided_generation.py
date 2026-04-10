# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""
Test Foundation Models guided generation
"""

import pytest
import tester_schemas.schemas as tester_schemas
from conftest import assert_schema_properties
from tester_schemas.validate_schemas import (
    validate_age,
    validate_cat,
    validate_hedgehog,
    validate_newsletter,
    validate_person,
    validate_pet_club,
    validate_shelter,
)

import apple_fm_sdk as fm


@pytest.mark.asyncio
async def test_age_generable(model):
    """Test using Age Generable class."""
    print("\n=== Testing Age Generable ===")

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use guided generation with Age
    result = await session.respond(
        "Generate an age for an elderly cat",
        generating=tester_schemas.Age,
    )
    assert type(result) is tester_schemas.Age, f"✗ Invalid generated content type: {type(result)}"

    # Validate the structured results
    print(f"✓ Got structured result: {type(result).__name__}")
    print(f"✓ Got generated content: {result}")

    # Validate using the validator
    validate_age(result)
    print("✓ Age validation passed")


@pytest.mark.asyncio
async def test_cat_generable(model):
    """Test using Cat Generable class."""
    print("\n=== Testing Cat Generable ===")

    session = fm.LanguageModelSession(
        "You are a helpful assistant that generates cat profiles.", model=model
    )

    # Use guided generation with GenerableCat
    result = await session.respond(
        "Generate a friendly kitten who loves playing fetch.",
        generating=tester_schemas.Cat,
    )
    assert type(result) is tester_schemas.Cat, f"✗ Invalid generated content type: {type(result)}"

    # Validate the structured results
    print(f"✓ Got structured result: {type(result).__name__}")
    print(f"✓ Got generated content: {result}")

    # Validate using the validator
    validate_cat(result)
    print("✓ Cat validation passed")


@pytest.mark.asyncio
async def test_hedgehog_generable(model):
    """Test using Hedgehog Generable class with complex generation guides."""
    print("\n=== Testing Hedgehog Generable: complex guides (range, anyOf, constant, count) ===")
    schema: fm.GenerationSchema = tester_schemas.Hedgehog.generation_schema()
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"
    print(f"GenerationSchema is {schema.type_class}: {schema.description}")

    # Check schema correctly converts to Foundation Models Generable JSON Schema
    assert_schema_properties(schema, "Hedgehog", ["name", "age", "favoriteFood", "home", "hobbies"])

    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate a hedgehog named Sonic who is 3 years old and loves carrots",
        generating=tester_schemas.Hedgehog,
    )
    print(f"✓ Got generated content: {generated_content}")
    assert type(generated_content) is tester_schemas.Hedgehog, (
        f"✗ Invalid generated content type: {type(generated_content)}"
    )

    # Validate using the validator
    validate_hedgehog(generated_content)
    print("✓ Hedgehog validation passed")


@pytest.mark.asyncio
async def test_person_generable(model):
    """Test using Person Generable class with nested Generable."""
    print("\n=== Testing Person Generable: nested Generable ===")
    assert isinstance(tester_schemas.Person, fm.Generable), "Person is not Generable"
    schema: fm.GenerationSchema = tester_schemas.Person.generation_schema()
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"

    # Check schema correctly converts to Foundation Models Generable JSON Schema
    schema.to_dict()

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate an elderly inn keeper character who has 3 children and no grandchildren",
        generating=tester_schemas.Person,
    )
    print(f"✓ Got generated content: {generated_content}")
    assert type(generated_content) is tester_schemas.Person, (
        f"✗ Invalid generated content type: {type(generated_content)}"
    )

    # Validate using the validator
    validate_person(generated_content)
    print("✓ Person validation passed")

    # Prompt-specific validation: the prompt asks for 3 children
    assert len(generated_content.children) == 3, (
        f"✗ Generated {len(generated_content.children)} children but expected 3"
    )
    print("✓ Correctly generated 3 children as requested")


@pytest.mark.asyncio
async def test_shelter_generable(model):
    """Test using Shelter Generable class with nested Cat objects."""
    print("\n=== Testing Shelter Generable: array of nested objects ===")
    schema: fm.GenerationSchema = tester_schemas.Shelter.generation_schema()
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"
    print(f"GenerationSchema is {schema.type_class}: {schema.description}")

    # Check schema correctly converts to Foundation Models Generable JSON Schema
    assert_schema_properties(schema, "Shelter", ["cats"])

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly
    generated_content = await session.respond(
        "Generate a shelter with 3 cats: a shorthair named Whiskers, a longhair named Fluffy, and a hairless named Sphinx",
        generating=tester_schemas.Shelter,
    )
    print(f"✓ Got generated content: {generated_content}")
    assert type(generated_content) is tester_schemas.Shelter, (
        f"✗ Invalid generated content type: {type(generated_content)}"
    )

    # Validate using the validator
    validate_shelter(generated_content)
    print("✓ Shelter validation passed")


@pytest.mark.asyncio
async def test_pet_club_generable(model):
    """Test using PetClub Generable class with multiple nested object types."""
    print("\n=== Testing PetClub Generable: Complex schema with multiple nested object types ===")
    schema: fm.GenerationSchema = tester_schemas.PetClub.generation_schema()
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"
    print(f"GenerationSchema is {schema.type_class}: {schema.description}")

    # Check schema correctly converts to Foundation Models Generable JSON Schema
    assert_schema_properties(
        schema,
        "PetClub",
        ["members", "cats", "hedgehogs", "otherPets", "presidentName"],
    )

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly with a comprehensive prompt
    generated_content = await session.respond(
        "Generate a pet club with 2 members (Alice age 25 and Bob age 30), "
        "2 cats (a shorthair named Mittens and a longhair named Fluffy), "
        "1 hedgehog (named Spike who is 5 years old and loves carrots), "
        "2 other pets (a parrot and a turtle), "
        "and Alice as the president",
        generating=tester_schemas.PetClub,
    )
    print(f"✓ Got generated content: {generated_content}")
    assert type(generated_content) is tester_schemas.PetClub, (
        f"✗ Invalid generated content type: {type(generated_content)}"
    )

    # Validate using the validator
    validate_pet_club(generated_content)
    print("✓ Pet club validation passed")


@pytest.mark.asyncio
async def test_newsletter_generable(model):
    """Test using ShelterNewsletter Generable class with optional fields and nested objects."""
    print("\n=== Testing ShelterNewsletter Generable: optional fields and nested objects ===")
    schema: fm.GenerationSchema = tester_schemas.ShelterNewsletter.generation_schema()
    assert isinstance(schema, fm.GenerationSchema), "Invalid schema"
    print(f"GenerationSchema is {schema.type_class}: {schema.description}")

    # Check schema correctly converts to Foundation Models Generable JSON Schema
    assert_schema_properties(
        schema,
        "ShelterNewsletter",
        [
            "title",
            "topic",
            "sponsor",
            "issueNumber",
            "tags",
            "featuredCats",
            "featuredHedgehog",
            "featuredStaff",
        ],
    )

    # Get the default model
    session = fm.LanguageModelSession(model=model)

    # Use the schema directly with a comprehensive prompt
    generated_content = await session.respond(
        "Generate a newsletter featuring senior cats available for adoption! \
        cats available for adoption! \
        - Do not mention any Hedgehogs. \
        - Mention 3 staff members who love senior cats \
        - This article does NOT have a sponsor",
        generating=tester_schemas.ShelterNewsletter,
    )
    print(f"✓ Got generated content: {generated_content}")
    assert type(generated_content) is tester_schemas.ShelterNewsletter, (
        f"✗ Invalid generated content type: {type(generated_content)}"
    )

    # Validate using the validator
    validate_newsletter(generated_content)
    print("✓ Newsletter validation passed")

    # Prompt-specific validations
    assert generated_content.sponsor is None, "✗ Sponsor should be None as per prompt"
    print("✓ Correctly omitted sponsor as requested")

    assert generated_content.featuredHedgehog is None, (
        "✗ Featured hedgehog should be None as per prompt"
    )
    print("✓ Correctly omitted hedgehog as requested")

    if generated_content.featuredStaff is not None:
        assert len(generated_content.featuredStaff) == 3, (
            f"✗ Featured staff count incorrect: {len(generated_content.featuredStaff)} instead of 3"
        )
    print("✓ Correctly included 3 staff members as requested")
