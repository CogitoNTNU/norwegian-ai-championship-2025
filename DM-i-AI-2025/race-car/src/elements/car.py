import pygame
from typing import Optional
from ..mathematics.vector import Vector 
from .road import Lane   
from ..mathematics.randomizer import random_number

class Car:
    def __init__(self, color: str, velocity: Vector, lane: Optional[Lane] = None, target_height: int = 40):
        """
        Initialize a Car object.

        :param color: The color of the car ('blue', 'yellow', 'red').
        :param velocity: The velocity of the car as a Vector.
        :param lane: The lane the car is in (optional).
        """
        self.color = color
        self.velocity = velocity
        self.lane = lane
        self.x = 0
        self.y = 0
        self.sprite = self.load_sprite(f"public/assets/{color}car.png", target_height)

    def update(self, ego: 'Car'):
        """
        Update the car's position based on its velocity and the ego car's velocity.

        :param ego: The ego car (reference car).
        """
        if self == ego:
            self.y += self.velocity.y
            return
        self.x += self.velocity.x - ego.velocity.x 
        self.y += self.velocity.y
        rn = random_number() - 0.5
        velocity_change = 0.1 * rn + 1 
        self.velocity.x = velocity_change * self.velocity.x


    def slow_down(self, amount: float = 0.1):
        """
        Reduce the car's velocity in the x-direction.

        :param amount: The amount to reduce the velocity by (default is 1).
        """
        self.velocity.x -= amount
        if self.velocity.x < 0:
            self.velocity.x = 0

    def speed_up(self, amount: float = 0.1):
        """
        Increase the car's velocity in the x-direction.

        :param amount: The amount to increase the velocity by (default is 1).
        """
        self.velocity.x += amount

    def turn(self, amount: float = 0.1):
        """
        Adjust the car's velocity in the y-direction.

        :param amount: The amount to adjust the velocity by (default is 1).
        """
        self.velocity.y += amount

    def load_sprite(self, path: str, target_height: int) -> pygame.Surface:
        """
        Load the car's sprite from the given file path.

        :param path: The file path to the sprite image.
        :param target_height: The target height to scale the sprite to.
        :return: The loaded sprite as a pygame.Surface.
        """
        try:
            sprite = pygame.image.load(path)
            aspect_ratio = sprite.get_width() / sprite.get_height()
            new_height = target_height
            new_width = int(aspect_ratio * new_height)
            sprite = pygame.transform.scale(sprite, (new_width, new_height))
            return sprite
        except pygame.error as e:
            print(f"Error loading sprite: {e}")
            return pygame.Surface((target_height, target_height))  # Return a placeholder surface if loading fails

    @property
    def rect(self):
        """Return the pygame.Rect representing the car's current position and size."""
        if self.sprite:
            return pygame.Rect(
                int(self.x),
                int(self.y),
                self.sprite.get_width(),
                self.sprite.get_height()
            )
        else:
            # Default size if sprite is missing
            return pygame.Rect(int(self.x), int(self.y), 50, 50)

    def get_bounds(self) -> pygame.Rect:
        """
        Returns the bounding rectangle of the car for collision/sensor purposes.
        """
        return self.rect