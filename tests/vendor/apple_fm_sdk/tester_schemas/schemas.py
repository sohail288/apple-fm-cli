# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import apple_fm_sdk as fm
from typing import List, Optional

# These are the exact same schemas as in tests/tester_schemas/schemas.swift,
# but expressed in Python syntax. They are used to test schema generation
# and parsing and ensure parity between the Swift and Python schemas.


# Simple example
@fm.generable()
class Age:
    years: int
    months: int


# Simplest example with a referenced generable type
@fm.generable()
class Cat:
    name: str
    age: Age
    profile: str


# Added complexity of Guides
@fm.generable()
class Hedgehog:
    name: str = fm.guide(description="A cute old-timey name")
    age: Age = fm.guide(description="The hedgehog's age, at most 8 years")
    favoriteFood: str = fm.guide(
        description="The hedgehog's favorite food", anyOf=["carrot", "turnip", "leek"]
    )
    home: str = fm.guide(constant="a hedge")
    hobbies: list[str] = fm.guide(description="The hedgehog's hobbies", count=3)


# An example that wraps a different generable type in a collection
@fm.generable()
class Shelter:
    cats: list["Cat"]


# This is a more complex example because it's nested and self-referential
@fm.generable()
class Person:
    age: Optional[int] = fm.guide(range=(18, 100))
    children: List["Person"] = fm.guide(
        description="The person's children", max_items=3
    )
    name: str = fm.guide(description="The person's name")


# An extra-complex example that references multiple generable types
@fm.generable()
class PetClub:
    members: list["Person"]
    cats: list["Cat"]
    hedgehogs: list["Hedgehog"]
    otherPets: list[str]
    presidentName: str = fm.guide(
        description="Should be the name of one of the members"
    )


# An extra-complex example with optional properties
@fm.generable()
class ShelterNewsletter:
    title: str
    topic: str
    sponsor: Optional[str] = fm.guide(
        description="A local company that's sponsoring this newsletter, if applicable"
    )
    issueNumber: Optional[int]
    tags: Optional[List[str]] = fm.guide(
        description="Search keywords for this newsletter"
    )
    featuredCats: Optional[List[Cat]]
    featuredHedgehog: Optional[Hedgehog]
    featuredStaff: Optional[List["Person"]]
