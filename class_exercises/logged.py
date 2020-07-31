# Name: Ofir Cohen
# ID: 312255847
# Date: 03/11/2019

def logged(func):
    def wrapper(*args):
        res = func(*args)
        print("you called {}{}\nit returned {}".format(func.__name__, args, res))
        return res
    return wrapper


@logged
def func(*args):
    return 3 + len(args)


if __name__ == "__main__":
    print(func(4, 4, 4))