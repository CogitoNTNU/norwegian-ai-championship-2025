import pygame
from typing import List, Dict
from .wall import Wall  
from ..mathematics.randomizer import random_number

class Lane:
    def __init__(self, y_start: float, y_end: float, name: str):
        self.y_start = y_start
        self.y_end = y_end
        self.name = name


class Road:
    def __init__(self, width: int, height: int, lanes: int):
        self._width = width
        self._height = height
        self._lane_count = lanes

        # Margins at top and bottom
        self._margin = 40
        self._line_height = 10  # Sideline and lane line thickness

        # Calculate usable height for lanes
        usable_height = self._height - 2 * self._margin
        self.lane_height = usable_height / lanes

        self.lanes: List[Lane] = []
        self.walls: List[Wall] = []

        self.surface = pygame.Surface((width, height))
        self.build_background()
        self.build_lanes(self.lane_height)

        self.y_start = self.lanes[0].y_start
        self.y_end = self.lanes[-1].y_end

        self.walls = [
            Wall(0, 0, self._width, self._margin),  # Top wall
            Wall(0, self._height - self._margin, self._width, self._margin)  # Bottom wall
        ]

    def first_lane(self) -> Lane:
        return self.lanes[0]

    def middle_lane(self) -> Lane:
        return self.lanes[len(self.lanes) // 2]

    def last_lane(self) -> Lane:
        return self.lanes[-1]

    def random_lane(self) -> Lane:
        return self.lanes[int(random_number() * len(self.lanes))] 

    def build_background(self):
        """
        Build the road background.
        """
        self.surface.fill((40, 44, 52))  # Dark gray background

    def build_lanes(self, lane_height: float):
        self.lanes.clear()
        for i in range(self._lane_count):
            y_start = self._margin + i * lane_height
            y_end = y_start + lane_height
            self.lanes.append(Lane(y_start, y_end, f"Lane {i+1}"))

        self.build_sidelines()
        self.build_middle_lines()

    def build_sidelines(self):
        # Draw top and bottom sidelines
        self.draw_line(0, self.lanes[0].y_start, self._width, self._line_height, (255, 255, 255))
        self.draw_line(0, self.lanes[-1].y_end - self._line_height, self._width, self._line_height, (255, 255, 255))

    def build_middle_lines(self):
        # Draw dashed lines between lanes
        for lane in self.lanes[1:]:
            self.draw_line(0, lane.y_start, self._width, self._line_height, (255, 255, 255))

    def draw_line(self, x: int, y: int, width: int, height: int, color: tuple):
        pygame.draw.rect(self.surface, color, pygame.Rect(x, int(y), width, int(height)))

    def get_lane_height(self):
        return self.lane_height