# Name: Ofir Cohen
# ID: 312255847
# Date: 27/10/2019

Rot13 = {
    'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u',
    'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c',
    'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
    'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S',
    'G':'T', 'H':'U', 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A',
    'O':'B', 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I',
    'W':'J', 'X':'K', 'Y':'L', 'Z':'M'
}


def cipher(statement):
    """ Gets a statement that is coded and deciphers it. """
    deciphered = ""
    for word in statement:
        if Rot13.get(word):
            deciphered += Rot13.get(word)
        else:
            deciphered += word
    
    return deciphered

if __name__ == "__main__":
    word = "Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!"
    # word = input("Enter a cipher statement: ")
    print(cipher(word))