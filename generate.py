import random
import json
def generate_type1(number,length):
    """
    Generates subtraction problems with alternating digits (e.g., 372918 - 729372).

    Args:
        number (int): Number of problem pairs to generate.
        length (int): Length of each number.

    Returns:
        list[dict]: A list of dictionaries containing 'x' and 'y' as operands.
    """
    data = []
    judge = True
    for i in range(0,number):
        value = dict()
        value['x'] = ''
        value['y'] = ''
        for j in range(0,length):
            x = random.randint(int(j == 0),9)
            y = random.randint(int(j == 0),9)
            if (x >= y) == judge:
                value['x'] += str(x)
                value['y'] += str(y)
            else:
                value['x'] += str(y)
                value['y'] += str(x)
            judge = not judge
        data.append(value)
    return data

def generate_type2(number,length):
    """
    Generates subtraction problems with alternating digits, ensuring no repeated digits.

    Args:
        number (int): Number of problem pairs to generate.
        length (int): Length of each number.

    Returns:
        list[dict]: A list of dictionaries containing 'x' and 'y' as operands.
    """
    data = []
    judge = True
    for i in range(0,number):
        value = dict()
        value['x'] = ''
        value['y'] = ''
        for j in range(0,length):
            x = random.randint(int(j == 0),9)
            y = random.randint(int(j == 0),9)
            while x == y:
                x = random.randint(int(j == 0),9)
                y = random.randint(int(j == 0),9)

            if (x > y) == judge:
                value['x'] += str(x)
                value['y'] += str(y)
            else:
                value['x'] += str(y)
                value['y'] += str(x)
            judge = not judge
        data.append(value)
    return data

def generate_type3(number,length):
    """
    Generates subtraction problems designed to produce frequent borrowing.

    Args:
        number (int): Number of problem pairs to generate.
        length (int): Length of each number.

    Returns:
        list[dict]: A list of dictionaries containing 'x' and 'y' as operands.
    """
    data = []
    for i in range(0,number):
        value = dict()
        value['x'] = ''
        value['y'] = ''
        for j in range(0,length):
            x = random.randint(int(j == 0),9)
            y = random.randint(int(j == 0),9)
            if j != length - 1:
                value['x'] += str(x)
                value['y'] += str(x)
            else:
                a = max(x,y)
                b = min(x,y)
                value['x'] += str(b)
                value['y'] += str(a)
        data.append(value)
    return data

def generate_type4(number,length_min,length_max=5):
    """
    Generates addition problems with random lengths within specified limits.

    Args:
        number (int): Number of problem pairs to generate.
        length_min (int): Minimum length of each number.
        length_max (int): Maximum length of each number.

    Returns:
        list[dict]: A list of dictionaries containing 'x' and 'y' as operands.
    """
    data = []
    judge = True
    for i in range(0,number):
        mid = random.randint(length_min,length_max)
        value = dict()
        value['x'] = ''
        value['y'] = ''
        for j in range(0,mid):
            x = random.randint(int(j == 0),9)
            y = random.randint(int(j == 0),9)
            a = max(x,y)
            b = min(x,y)
            if a == b and a != 9:
                a += 1
            elif a == b and a == 9:
                b -= 1
            value['x'] += str(b)
            value['y'] += str(a)
        for k in range(mid - length_max + 1,1):
            y = random.randint(int(k == 0),9)
            value['y'] = str(y) + value['y']
        data.append(value)
    return data

def generate_type5(number,length_min,length_max):
    """
    Generates addition problems with random lengths within specified limits.

    Args:
        number (int): Number of problem pairs to generate.
        length_min (int): Minimum length of each number.
        length_max (int): Maximum length of each number.

    Returns:
        list[dict]: A list of dictionaries containing 'x' and 'y' as operands.
    """
    data = []
    judge = True
    for i in range(0,number):
        a = random.randint(length_min,length_max)
        b = random.randint(length_min,length_max)
        mid = min(a,b)
        length = max(a,b)

        value = dict()
        value['x'] = ''
        value['y'] = ''
        for j in range(0,mid):
            x = random.randint(int(j == 0),9 - int(j == 0))
            y = 9 - x
            if j == mid - 1 and x != 9:
                x += 1
            elif j == mid - 1 and x == 9:
                y += random.randint(0,9)
            value['x'] += str(x)
            value['y'] += str(y)
        for k in range(mid - length + 1,1):
            y = random.randint(int(k == 0),9)
            value['y'] = str(y) + value['y']
        data.append(value)
    return data

def total_type(data,symb = '-',x=0,y=9):
    """
    Generates mixed datasets with different orders for operands.

    Args:
        data (list[dict]): List of operand pairs.
        symb (str): Operator to include in the problems.
        x (int): Lower bound for randomness.
        y (int): Upper bound for randomness.

    Returns:
        list[list]: Mixed datasets with operands and operator.
    """
    my_list = []
    for i in data:
        value = ''
        a = i['x']
        b = i['y']
        if random.randint(x,y) % 2:
            value = a + symb + b
        else:
            value = b + symb + a
        temp_list = []
        for i in value:
            temp_list.append(i)
        my_list.append(temp_list)
    return my_list

data = generate_type5(100,2,4)
my = total_type(data,'+')

file_name = 'diff_sum.json'
with open(file_name,'w') as f:
    json.dump(my,f)

with open(file_name,'r') as f:
    content = json.load(f)

for i in content:
    print(i)
