# схема сутностей для корпусу новин про емо-реп
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "Person",        # артист, продюсер, родич
    "Organization",  # лейбл, колектив, компанія, фонд
    "Release",       # альбом, мікстейп, сингл, фільм
    "Place",         # місто, штат, заклад
    "Event",         # смерть, суд, тур, фестиваль
    "Chart",         # Billboard 200, Billboard Hot 100
]

RelationType = Literal[
    "BORN_IN",            # Person -> Place
    "DIED_IN",            # Person -> Place
    "MEMBER_OF",          # Person -> Organization (колектив)
    "SIGNED_TO",          # Person -> Organization (лейбл)
    "RELEASED",           # Person/Organization -> Release
    "FEATURED_ON",        # Person -> Release (гість, співавтор треку)
    "PRODUCED",           # Person/Organization -> Release
    "COLLABORATED_WITH",  # Person -> Person
    "SAMPLED",            # Release -> Release/Person
    "CHARTED_ON",         # Release -> Chart
    "PARTICIPATED_IN",    # Person -> Event
    "FOUNDED",            # Person -> Organization
    "RELATED_TO",         # родинні зв'язки, Person -> Person
]


class Entity(BaseModel):
    name: str = Field(description="Канонічна назва: сценічне ім'я артиста ('Lil Peep', а не 'Густав Ар'), назва релізу без лапок, у називному відмінку")
    type: EntityType
    description: str = Field(description="Короткий опис за текстом, до 15 слів")


class Fact(BaseModel):
    subject: str = Field(description="Канонічна назва суб'єкта, як у entities")
    subject_type: EntityType
    predicate: RelationType
    object: str = Field(description="Канонічна назва об'єкта, як у entities")
    object_type: EntityType
    year: int = Field(description="Рік факту з тексту або 0")
    evidence: str = Field(description="Дослівне речення з тексту, що підтверджує факт")


class Extraction(BaseModel):
    entities: list[Entity]
    facts: list[Fact]
