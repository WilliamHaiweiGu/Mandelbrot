from PIL import Image
from tqdm import tqdm

if __name__=="__main__":
    with open("out.dat", "rb") as file:
        # Read the entire file
        data = file.read()
    width=65536
    height=len(data)//width
    img = Image.frombytes(data=data, mode='L', size=(width,height))
    del data
    pixels = img.load()
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    for x in tqdm(range(width)):
        for y in range(height):
            if pixels[x, y] == 0:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    del pixels
    if min_x > max_x or min_y > max_y:
        print("No black pixels found.")
    else:
        img=img.crop((min_x, min_y, max_x, max_y))

    w1, h1 = img.size
    out=Image.new('L',(w1,2*h1-1))
    out.paste(img, (0, 0))
    out.paste(img.crop((0, 0, w1, h1-1)),h1)
    out.save('out.png')