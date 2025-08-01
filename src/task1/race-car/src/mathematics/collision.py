import pygame
from .vector import Vector  # Assuming a Vector class exists
from typing import Optional, List

class Line:
    def __init__(self, start: Vector, end: Vector):
        self.start = start
        self.end = end

def intersects(a: pygame.Rect, b: pygame.Rect) -> bool:
    """
    Check if two rectangles intersect.

    :param a: The first rectangle.
    :param b: The second rectangle.
    :return: True if the rectangles intersect, False otherwise.
    """
    return (
        a.x + a.width > b.x and
        a.x < b.x + b.width and
        a.y + a.height > b.y and
        a.y < b.y + b.height
    )

def get_intersection_point(v: Line, u: Line) -> Optional[Vector]:
    """
    Get the intersection point of two lines, if it exists.

    :param v: The first line.
    :param u: The second line.
    :return: The intersection point as a Vector, or None if no intersection exists.
    """
    b = v.end.sub(v.start)
    d = u.end.sub(u.start)
    bxd = b.cross(d)

    if bxd == 0:
        # Parallel lines, no intersection
        return None

    c = u.start.sub(v.start)
    t = (c.x * d.y - c.y * d.x) / bxd

    if t < 0 or t > 1:
        return None

    w = (c.x * b.y - c.y * b.x) / bxd

    if w < 0 or w > 1:
        return None

    intersection = v.start.add(b.scale(t))
    return intersection

def get_lines_of_rectangle(r: pygame.Rect) -> List[Line]:
    """
    Get the lines (edges) of a rectangle.

    :param r: The rectangle.
    :return: A list of Line objects representing the edges of the rectangle.
    """
    bot_left = Vector(r.left, r.bottom)
    bot_right = Vector(r.right, r.bottom)
    top_left = Vector(r.left, r.top)
    top_right = Vector(r.right, r.top)

    return [
        Line(bot_left, bot_right),   # Bottom
        Line(bot_right, top_right),  # Right
        Line(top_left, top_right),   # Top
        Line(top_left, bot_left),    # Left
    ]