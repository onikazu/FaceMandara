from PIL import Image, ImageDraw, ImageFont

class Line:
    def __init__(self, x0=0, y0=0, x1=0, y1=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.cross_x = 0
        self.cross_y = 0

    def draw_line(self, frame, rect):
        """
        rect = (left, top, right, bottom)
        """
        print("x0, y0, x1, y1", self.x0, self.y0, self.x1, self.y1)
        print("cross_x, cross_y", self.cross_x, self.cross_y)
        left, top, right, bottom = rect
        # 枠内では線を表示しないようにしてやる
        if top<self.y1<bottom and left<self.x1<right:
            return
        # フレームと線の交点
        if (self.x1 >= right or self.x1 <= left or self.y1 <= top or self.y1 >= bottom) and self.cross_x == 0:
            self.cross_x = self.x1
            self.cross_y = self.y1
            return
        draw = ImageDraw.Draw(frame)
        draw.line((self.cross_x, self.cross_y, self.x1, self.y1), fill=(255, 255, 255), width=3)





    def setter(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
