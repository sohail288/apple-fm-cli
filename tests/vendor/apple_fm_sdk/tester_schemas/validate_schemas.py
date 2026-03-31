# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import apple_fm_sdk as fm
from typing import List, Any
import tester_schemas.schemas as tester_schemas


# Helper function to convert Generable to dict
def _generable_to_dict(obj: Any) -> Any:
    """
    Convert a Generable object to a dict for validation.
    Handles nested Generable objects and lists.
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [_generable_to_dict(item) for item in obj]
    elif isinstance(obj, fm.Generable):
        # It's a Generable object, convert to dict
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                result[key] = _generable_to_dict(value)
        return result
    else:
        return obj


def validate_hedgehog(hedgehog_data) -> bool:
    """
    Validate hedgehog generated content.

    Args:
        hedgehog_data: Either a GeneratedContent object or a dict containing hedgehog data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(hedgehog_data, tester_schemas.Hedgehog):
        hedgehog_data = _generable_to_dict(hedgehog_data)

    # Extract properties based on input type
    if isinstance(hedgehog_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        name = hedgehog_data.value(str, for_property="name")
        age = hedgehog_data.value(dict, for_property="age")
        favoriteFood = hedgehog_data.value(str, for_property="favoriteFood")
        home = hedgehog_data.value(str, for_property="home")
        hobbies = hedgehog_data.value(List[str], for_property="hobbies")
    elif isinstance(hedgehog_data, dict):
        # Extract from dict
        name = hedgehog_data.get("name")
        age = hedgehog_data.get("age")
        favoriteFood = hedgehog_data.get("favoriteFood")
        home = hedgehog_data.get("home")
        hobbies = hedgehog_data.get("hobbies")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(hedgehog_data)}"
        )

    # Validate name
    assert isinstance(name, str), f"name must be string, got {type(name)}"
    assert len(name) > 0, "name must not be empty"

    # Validate age structure and constraints
    validate_age(age)

    # Validate favoriteFood enum constraint
    valid_foods = ["carrot", "turnip", "leek"]
    assert favoriteFood in valid_foods, (
        f"favoriteFood must be one of {valid_foods}, got '{favoriteFood}'"
    )

    # Validate home constant constraint
    assert home == "a hedge", f"home must be 'a hedge', got '{home}'"

    # Validate hobbies array constraints
    assert isinstance(hobbies, list), f"hobbies must be list, got {type(hobbies)}"
    assert len(hobbies) == 3, f"hobbies must have exactly 3 items, got {len(hobbies)}"
    for i, hobby in enumerate(hobbies):
        assert isinstance(hobby, str), f"hobbies[{i}] must be string, got {type(hobby)}"
        assert len(hobby) > 0, f"hobbies[{i}] must not be empty"

    # Return the validated data
    return True


def validate_person(person_data) -> bool:
    """
    Validate person generated content.

    Args:
        person_data: Either a GeneratedContent object or a dict containing person data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(person_data, tester_schemas.Person):
        person_data = _generable_to_dict(person_data)

    # Extract properties based on input type
    if isinstance(person_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        name = person_data.value(str, for_property="name")
        # Age is optional, so we need to check if it exists
        try:
            age = person_data.value(int, for_property="age")
        except (KeyError, AttributeError):
            age = None
        children = person_data.value(List[dict], for_property="children")
    elif isinstance(person_data, dict):
        # Extract from dict
        name = person_data.get("name")
        age = person_data.get("age")
        children = person_data.get("children")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(person_data)}"
        )

    # Validate name (required)
    assert isinstance(name, str), f"name must be string, got {type(name)}"
    assert len(name) > 0, "name must not be empty"

    # Validate age (optional, but if present must meet constraints)
    if age is not None:
        assert isinstance(age, int), f"age must be int, got {type(age)}"
        assert 18 <= age <= 100, f"age must be between 18 and 100, got {age}"

    # Validate children (required)
    assert isinstance(children, list), f"children must be list, got {type(children)}"
    assert len(children) <= 3, (
        f"children must have at most 3 items, got {len(children)}"
    )
    for i, child in enumerate(children):
        assert isinstance(child, dict), f"children[{i}] must be dict, got {type(child)}"
        # Recursively validate each child as a Person
        validate_person(child)

    return True


def validate_cat(cat_data) -> bool:
    """
    Validate cat generated content.

    Args:
        cat_data: Either a GeneratedContent object or a dict containing cat data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(cat_data, tester_schemas.Cat):
        cat_data = _generable_to_dict(cat_data)

    # Extract properties based on input type
    if isinstance(cat_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        name = cat_data.value(str, for_property="name")
        age = cat_data.value(dict, for_property="age")
        profile = cat_data.value(str, for_property="profile")
    elif isinstance(cat_data, dict):
        # Extract from dict
        name = cat_data.get("name")
        age = cat_data.get("age")
        profile = cat_data.get("profile")
    else:
        raise AssertionError(f"Expected GeneratedContent or dict, got {type(cat_data)}")

    # Validate name (required)
    assert isinstance(name, str), f"name must be string, got {type(name)}"
    assert len(name) > 0, "name must not be empty"

    # Validate age structure and constraints (required)
    validate_age(age)

    # Validate profile (required)
    assert isinstance(profile, str), f"profile must be string, got {type(profile)}"
    assert len(profile) > 0, "profile must not be empty"

    return True


def validate_age(age_data) -> bool:
    """
    Validate age generated content.

    Args:
        age_data: Either a GeneratedContent object or a dict containing age data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(age_data, tester_schemas.Age):
        age_data = _generable_to_dict(age_data)

    # Extract properties based on input type
    if isinstance(age_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        years = age_data.value(int, for_property="years")
        months = age_data.value(int, for_property="months")
    elif isinstance(age_data, dict):
        # Extract from dict
        years = age_data.get("years")
        months = age_data.get("months")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(age_data)}: {age_data}"
        )

    # Validate years (required)
    assert isinstance(years, int), f"years must be int, got {type(years)}"

    # Validate months (required)
    assert isinstance(months, int), f"months must be int, got {type(months)}"

    return True


def validate_shelter(shelter_data) -> bool:
    """
    Validate shelter generated content.

    Args:
        shelter_data: Either a GeneratedContent object or a dict containing shelter data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(shelter_data, tester_schemas.Shelter):
        shelter_data = _generable_to_dict(shelter_data)

    # Extract properties based on input type
    if isinstance(shelter_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        cats = shelter_data.value(List[dict], for_property="cats")
    elif isinstance(shelter_data, dict):
        # Extract from dict
        cats = shelter_data.get("cats")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(shelter_data)}"
        )

    # Validate cats (required)
    assert isinstance(cats, list), f"cats must be list, got {type(cats)}"
    for i, cat in enumerate(cats):
        assert isinstance(cat, dict), f"cats[{i}] must be dict, got {type(cat)}"
        # Recursively validate each cat
        validate_cat(cat)

    return True


def validate_pet_club(pet_club_data) -> bool:
    """
    Validate pet club generated content.

    Args:
        pet_club_data: Either a GeneratedContent object or a dict containing pet club data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(pet_club_data, tester_schemas.PetClub):
        pet_club_data = _generable_to_dict(pet_club_data)

    # Extract properties based on input type
    if isinstance(pet_club_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        members = pet_club_data.value(List[dict], for_property="members")
        cats = pet_club_data.value(List[dict], for_property="cats")
        hedgehogs = pet_club_data.value(List[dict], for_property="hedgehogs")
        otherPets = pet_club_data.value(List[str], for_property="otherPets")
        presidentName = pet_club_data.value(str, for_property="presidentName")
    elif isinstance(pet_club_data, dict):
        # Extract from dict
        members = pet_club_data.get("members")
        cats = pet_club_data.get("cats")
        hedgehogs = pet_club_data.get("hedgehogs")
        otherPets = pet_club_data.get("otherPets")
        presidentName = pet_club_data.get("presidentName")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(pet_club_data)}"
        )

    # Validate members (required)
    assert isinstance(members, list), f"members must be list, got {type(members)}"
    for i, member in enumerate(members):
        assert isinstance(member, dict), (
            f"members[{i}] must be dict, got {type(member)}"
        )
        # Recursively validate each member as a Person
        validate_person(member)

    # Validate cats (required)
    assert isinstance(cats, list), f"cats must be list, got {type(cats)}"
    for i, cat in enumerate(cats):
        assert isinstance(cat, dict), f"cats[{i}] must be dict, got {type(cat)}"
        # Recursively validate each cat
        validate_cat(cat)

    # Validate hedgehogs (required)
    assert isinstance(hedgehogs, list), f"hedgehogs must be list, got {type(hedgehogs)}"
    for i, hedgehog in enumerate(hedgehogs):
        assert isinstance(hedgehog, dict), (
            f"hedgehogs[{i}] must be dict, got {type(hedgehog)}"
        )
        # Recursively validate each hedgehog
        validate_hedgehog(hedgehog)

    # Validate otherPets (required)
    assert isinstance(otherPets, list), f"otherPets must be list, got {type(otherPets)}"
    for i, pet in enumerate(otherPets):
        assert isinstance(pet, str), f"otherPets[{i}] must be string, got {type(pet)}"

    # Validate presidentName (required)
    assert isinstance(presidentName, str), (
        f"presidentName must be string, got {type(presidentName)}"
    )
    assert len(presidentName) > 0, "presidentName must not be empty"

    return True


def validate_newsletter(newsletter_data) -> bool:
    """
    Validate newsletter generated content.

    Args:
        newsletter_data: Either a GeneratedContent object or a dict containing newsletter data

    Returns:
        bool: True if validation passes

    Raises:
        AssertionError: If any validation fails
    """
    # Convert Generable to dict for easier validation
    if isinstance(newsletter_data, tester_schemas.ShelterNewsletter):
        newsletter_data = _generable_to_dict(newsletter_data)

    # Extract properties based on input type
    if isinstance(newsletter_data, fm.GeneratedContent):
        # Extract all required properties from GeneratedContent
        title = newsletter_data.value(str, for_property="title")
        topic = newsletter_data.value(str, for_property="topic")
        # Optional fields
        try:
            sponsor = newsletter_data.value(str, for_property="sponsor")
        except (KeyError, AttributeError):
            sponsor = None
        try:
            issueNumber = newsletter_data.value(int, for_property="issueNumber")
        except (KeyError, AttributeError):
            issueNumber = None
        try:
            tags = newsletter_data.value(List[str], for_property="tags")
        except (KeyError, AttributeError):
            tags = None
        try:
            featuredCats = newsletter_data.value(
                List[dict], for_property="featuredCats"
            )
        except (KeyError, AttributeError):
            featuredCats = None
        try:
            featuredHedgehog = newsletter_data.value(
                dict, for_property="featuredHedgehog"
            )
        except (KeyError, AttributeError):
            featuredHedgehog = None
        try:
            featuredStaff = newsletter_data.value(
                List[dict], for_property="featuredStaff"
            )
        except (KeyError, AttributeError):
            featuredStaff = None
    elif isinstance(newsletter_data, dict):
        # Extract from dict
        title = newsletter_data.get("title")
        topic = newsletter_data.get("topic")
        sponsor = newsletter_data.get("sponsor")
        issueNumber = newsletter_data.get("issueNumber")
        tags = newsletter_data.get("tags")
        featuredCats = newsletter_data.get("featuredCats")
        featuredHedgehog = newsletter_data.get("featuredHedgehog")
        featuredStaff = newsletter_data.get("featuredStaff")
    else:
        raise AssertionError(
            f"Expected GeneratedContent or dict, got {type(newsletter_data)}"
        )

    # Validate title (required)
    assert isinstance(title, str), f"title must be string, got {type(title)}"
    assert len(title) > 0, "title must not be empty"

    # Validate topic (required)
    assert isinstance(topic, str), f"topic must be string, got {type(topic)}"
    assert len(topic) > 0, "topic must not be empty"

    # Validate sponsor (optional)
    if sponsor is not None:
        assert isinstance(sponsor, str), f"sponsor must be string, got {type(sponsor)}"

    # Validate issueNumber (optional)
    if issueNumber is not None:
        assert isinstance(issueNumber, int), (
            f"issueNumber must be int, got {type(issueNumber)}"
        )

    # Validate tags (optional)
    if tags is not None:
        assert isinstance(tags, list), f"tags must be list, got {type(tags)}"
        for i, tag in enumerate(tags):
            assert isinstance(tag, str), f"tags[{i}] must be string, got {type(tag)}"

    # Validate featuredCats (optional)
    if featuredCats is not None:
        assert isinstance(featuredCats, list), (
            f"featuredCats must be list, got {type(featuredCats)}"
        )
        for i, cat in enumerate(featuredCats):
            assert isinstance(cat, dict), (
                f"featuredCats[{i}] must be dict, got {type(cat)}"
            )
            # Recursively validate each cat
            validate_cat(cat)

    # Validate featuredHedgehog (optional)
    if featuredHedgehog is not None:
        assert isinstance(featuredHedgehog, dict), (
            f"featuredHedgehog must be dict, got {type(featuredHedgehog)}"
        )
        # Recursively validate the hedgehog
        validate_hedgehog(featuredHedgehog)

    # Validate featuredStaff (optional)
    if featuredStaff is not None:
        assert isinstance(featuredStaff, list), (
            f"featuredStaff must be list, got {type(featuredStaff)}"
        )
        for i, staff in enumerate(featuredStaff):
            assert isinstance(staff, dict), (
                f"featuredStaff[{i}] must be dict, got {type(staff)}"
            )
            # Recursively validate each staff member as a Person
            validate_person(staff)

    return True
