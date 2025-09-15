from utils import is_equiv

str1 = r"(\pi - 2) / \pi"
str2 = r"1-\frac{2}{\pi}"
print(is_equiv(str1, str2, verbose=True))