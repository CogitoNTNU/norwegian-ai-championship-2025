import math
from typing import List

class Vector:
    def __init__(self, x: float = 0, y: float = 0):
        """
        Initialize a Vector object.

        :param x: The x-coordinate of the vector.
        :param y: The y-coordinate of the vector.
        """
        self.x = x
        self.y = y

    @staticmethod
    def zero():
        """
        Create a zero vector.

        :return: A Vector with x and y set to 0.
        """
        return Vector(0, 0)

    @staticmethod
    def from_array(arr: List[float]):
        """
        Create a Vector from an array.

        :param arr: A list containing two numbers [x, y].
        :return: A Vector object.
        """
        return Vector(arr[0], arr[1])

    def to_array(self) -> List[float]:
        """
        Convert the Vector to an array.

        :return: A list containing [x, y].
        """
        return [self.x, self.y]

    def clone(self):
        """
        Create a copy of the Vector.

        :return: A new Vector with the same x and y values.
        """
        return Vector(self.x, self.y)

    def add(self, v):
        """
        Add another Vector or a scalar to this Vector.

        :param v: A Vector or a scalar value.
        :return: A new Vector with the result of the addition.
        """
        if isinstance(v, Vector):
            return Vector(self.x + v.x, self.y + v.y)
        return Vector(self.x + v, self.y + v)

    def sub(self, v):
        """
        Subtract another Vector from this Vector.

        :param v: A Vector to subtract.
        :return: A new Vector with the result of the subtraction.
        """
        return Vector(self.x - v.x, self.y - v.y)

    def scale(self, v: float):
        """
        Scale the Vector by a scalar value.

        :param v: The scalar value to scale by.
        :return: A new Vector with the scaled values.
        """
        return Vector(self.x * v, self.y * v)

    def dot(self, v):
        """
        Compute the dot product with another Vector.

        :param v: Another Vector.
        :return: The dot product as a float.
        """
        return self.x * v.x + self.y * v.y

    def cross(self, v):
        """
        Compute the cross product with another Vector.

        :param v: Another Vector.
        :return: The cross product as a float.
        """
        return self.x * v.y - self.y * v.x

    def rotate(self, degrees: float):
        """
        Rotate the Vector by a given angle in degrees.

        :param degrees: The angle to rotate by, in degrees.
        :return: A new Vector with the rotated values.
        """
        radians = math.radians(degrees)
        return Vector(
            math.cos(radians) * self.x - math.sin(radians) * self.y,
            math.sin(radians) * self.x + math.cos(radians) * self.y
        )

    def distance(self, v):
        """
        Compute the distance between this Vector and another Vector.

        :param v: Another Vector.
        :return: The distance as a float.
        """
        return math.sqrt((v.x - self.x) ** 2 + (v.y - self.y) ** 2)