import requests
from bs4 import BeautifulSoup

# Replace with your published URL
url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"


class Image:
    def __init__(self, width, height):
        self.image = [[' ' for x in range(width)] for y in range(height)]

    def insert_pixel(self, pixel):
        x, ch, y = pixel
        self.image[int(y)][int(x)] = ch

    def show(self):
        for i in self.image:
            print(*i, sep='')

    @staticmethod
    def load_image(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find('table')
        rows = content.find_all('tr')[1:]
        width, height = Image.get_image_dimensions(rows)
        image = Image(width, height)
        for row in rows:
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols]
            image.insert_pixel(cols)
        return image

    @staticmethod
    def get_image_dimensions(rows):
        width = 0
        height = 0
        for row in rows:
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols]
            if int(cols[0])+1 > width:
                width = int(cols[0])+1
            if int(cols[2])+1 > height:
                height = int(cols[2])+1
        return (width, height)


image = Image.load_image(url)
image.show()
