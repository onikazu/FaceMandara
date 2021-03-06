"""
円形フレームに対応
"""

from PIL import Image, ImageDraw, ImageFont
import math


class Line:
    def __init__(self, x0=0, y0=0, x1=0, y1=0, movement_amount_x=0, movement_amount_y=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.cross_x = 0
        self.cross_y = 0
        self.movement_amount_x = movement_amount_x
        self.movement_amount_y = movement_amount_y

    def draw_line(self, frame, rect):
        """
        rect = (left, top, right, bottom)
        """
        print("x0, y0, x1, y1", self.x0, self.y0, self.x1, self.y1)
        print("cross_x, cross_y", self.cross_x, self.cross_y)
        left, top, right, bottom = rect
        circle_center_x = left + (right-left) / 2
        circle_center_y = top + (bottom-top) / 2
        radius = (right-left) / 2
        destination_x = circle_center_x + self.movement_amount_x
        destination_y = circle_center_y + self.movement_amount_y
        theta = math.atan2(destination_y-circle_center_y, destination_x-circle_center_x)
        self.cross_x = radius*math.cos(theta)+circle_center_x
        self.cross_y = radius*math.sin(theta)+circle_center_y

        if ((self.x1-circle_center_x)**2+(self.y1-circle_center_y))<radius**2:
            return




        # # 枠内では線を表示しないようにしてやる
        # if ((self.x1-circle_center_x)**2+(self.y1-circle_center_y))<radius**2:
        #     self.cross_x = self.x1
        #     self.cross_y = self.y1
        #     return
        # # フレームと線の交点
        # if ((self.x1-circle_center_x)**2+(self.y1-circle_center_y))>=radius**2 and self.cross_x == 0:
        #     self.cross_x = (self.cross_x+self.x1)/2
        #     self.cross_y = (self.cross_y+self.y1)/2
        #     return

        draw = ImageDraw.Draw(frame)
        draw.line((self.cross_x, self.cross_y, self.x1, self.y1), fill=(255, 255, 255), width=1)

    def setter(self, x0, y0, x1, y1, movement_amount_x, movement_amount_y):
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.movement_amount_x = int(movement_amount_x)
        self.movement_amount_y = int(movement_amount_y)
