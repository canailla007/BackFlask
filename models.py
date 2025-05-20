# models.py

from dataclasses import dataclass, asdict

@dataclass
class Car:
    model_id: int
    name: str
    velocity: str
    image_front: str
    image_rear: str
    fuel_type: str
    horsepower: str
    price: float
    passengers: int
    brand: str
    visible: bool = True

    def to_dict(self):
        return asdict(self)
