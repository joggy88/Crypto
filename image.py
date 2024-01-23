import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Replace 'btc.jpg' with your image file's path
base64_image = get_base64_of_bin_file('btc.jpg')
