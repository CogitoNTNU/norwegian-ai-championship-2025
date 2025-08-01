import pygame

class Wall:
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize a Wall object.

        :param x: The x-coordinate of the wall.
        :param y: The y-coordinate of the wall.
        :param width: The width of the wall.
        :param height: The height of the wall.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (21, 19, 23)  # Dark gray color

    def draw(self, surface: pygame.Surface):
        """
        Draw the wall on the given surface.

        :param surface: The pygame surface to draw the wall on.
        """
        pygame.draw.rect(surface, self.color, self.rect)

    def get_bounds(self) -> pygame.Rect:
        """
        Returns the bounding rectangle of the car for collision/sensor purposes.
        """
        return self.rect