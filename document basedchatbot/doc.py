import magic

def get_file_type(file_path):
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)

# Example usage
file_path = ''
file_type = get_file_type(file_path)
print(f'{file_type}')
