from PIL import Image, ImageDraw
import os


def draw_rectangle_pillow(image_path, target_width, target_height, pt1, pt2, pt3, pt4):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'output_pillow.jpg')

    img = Image.open(image_path)
    original_w, original_h = img.size

    scale_x = original_w / target_width
    scale_y = original_h / target_height

    orig_pt1 = (pt1[0] * scale_x, pt1[1] * scale_y)
    orig_pt3 = (pt3[0] * scale_x, pt3[1] * scale_y)
    orig_pt2 = (pt2[0] * scale_x, pt2[1] * scale_y)
    orig_pt4 = (pt4[0] * scale_x, pt4[1] * scale_y)

    draw = ImageDraw.Draw(img)
    draw.rectangle([orig_pt1, orig_pt2], outline="green", width=5)
    draw.rectangle([orig_pt3, orig_pt4], outline="green", width=5)

    img.save(output_path)
    print(f"File saved to: {output_path}")


draw_rectangle_pillow('tests/katrina.jpg', 178, 218, (65, 100), (85, 110),(65,75),(75,80))