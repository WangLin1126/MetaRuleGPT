from utils import *
# Base data consisting of letters used for encoding
base_data = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

class Dataset:
    def __init__(self,length,number):
        """
        Initializes the dataset for arithmetic operations.
        
        Args:
            length (int): The maximum length of numbers involved in operations.
            number (int): The base of the arithmetic system.
        """

        self.length = length
        self.number = number

        self.source = []
        self.target = []

    def mapping(self,i,j,symb = '+'):
        """
        Generates the source and target sequences for arithmetic expressions.
        
        Args:
            i (int): Length of the first operand.
            j (int): Length of the second operand.
            symb (str): The operator for the arithmetic operation ('+', '-', etc.).
        
        Returns:
            tuple: Source and target sequences for the arithmetic expression.
        """
        source = []
        target = []
        if symb == '-+' or symb == '--':
            for k in range(i + j + 4):
                source += [base_data[k]]
            source[0] = symb[0]
            source[i + 2] = symb[-1]
            x = i 
            y = j
            while x > -1 or y > -1:
                if x <= -1 and symb == '-+':
                    target = ['|'] + [source[i + y + 3]] + target
                elif y <= -1 and symb == '-+':
                    target = ['|'] + ['0','-'] +[source[x + 1]] + target
                elif x <= -1 and symb == '--':
                    target = ['|'] + [source[i + y + 3]] + target
                elif y <= -1 and symb == '--':
                    target = ['|'] + [source[x + 1]] + target
                else:
                    if symb == '-+':
                        target = ['|'] + [source[i + y + 3]] + ['-'] + [source[x + 1]] + target
                    else:
                        target = ['|'] + [source[i + y + 3]] + ['+'] + [source[x + 1]] + target        
                x -= 1
                y -= 1
            if symb == '--':
                target = ['e','-'] + target
            else:
                target[0] = 'e'
            source = ['s'] + source
            return source,target
        for k in range(i + j + 3):
            source += [base_data[k]]
        source[i + 1] = symb
        x = i
        y = j
        while x > -1 or y > -1:
            if x <= -1 and symb == '+':
                target = ['|'] + [source[i + y + 2]] + target
            elif y <= -1 and symb == '+':
                target = ['|'] + [source[x]] + target
            elif x <= -1 and symb == '-':
                target = ['|'] + ['0'] + ['-'] + [source[i + y + 2]] + target
            elif y <= -1 and symb == '-':
                target = ['|'] + [source[x]] + target
            else:
                target = ['|'] + [source[x]] + [symb] + [source[i + y + 2]] + target
            x -= 1
            y -= 1
        target[0] = 'e'
        source = ['s'] + source
        return source,target

    def align(self):
        """
        Aligns operands of various lengths and generates source-target pairs
        for arithmetic operations involving addition and subtraction.
        """
        for i in range(self.length):
            for j in range(self.length):
                source,target = self.mapping(i,j,'+')
                self.source.append(source)
                self.target.append(target)
                source,target = self.mapping(i,j,'-')
                self.source.append(source)
                self.target.append(target)
                source,target = self.mapping(i,j,'-+')
                self.source.append(source)
                self.target.append(target)
                source,target = self.mapping(i,j,'--')
                self.source.append(source)
                self.target.append(target)

    def add(self):
        """
        Generates basic addition and subtraction rules in the defined base system.
        """
        k = self.number
        for i in range(k):
            for j in range(k):
                source = ['s'] + [str(i)] + ['+'] + [str(j)]
                target = []
                if i + j >= k:
                    target = ['e'] + [str((i + j) // k)] + [str((i + j) % k)]
                else:
                    target = ['e'] + [str(i + j)]
                self.source.append(source)
                self.target.append(target)

                source = ['s'] + [str(i)] + ['-'] + [str(j)]
                target = []
                if i - j < 0:
                    target = ['e'] + ['-'] + [str(j - i)]
                else:
                    target = ['e'] + [str(i - j)]
                self.source.append(source)
                self.target.append(target)
        for i in range(k):
            self.source.append(['s',str(i)])
            self.target.append(['e',str(i)])
            if i >= 1:
                self.source.append(['s','-',str(i)])
                self.target.append(['e','-',str(i)])
                self.source.append(['s','-','+',str(i)])
                self.target.append(['e',str(i)])
        self.source.append(['s','-'])
        self.target.append(['e','-'])

        for i in range(1,k):
            source = ['s'] + ['*'] + ['-'] + [str(i)]
            target = ['e'] + [str(k - i)]
            self.source.append(source)
            self.target.append(target)

            source = ['s'] + ['?'] + ['-'] + [str(i)]
            target = ['e'] + [str(k - i - 1)]
            self.source.append(source)
            self.target.append(target)

            source = ['s'] + [str(i)] + ['-'] + ['*']
            if k - i - 1 == 0:
                target = ['e']  + [str(k - i - 1)]
            else:
                target = ['e'] + ['-'] + [str(k - i - 1)]
            target = ['e'] + ['-'] + [str(k - i)]
            self.source.append(source)
            self.target.append(target)
            source = ['s'] + [str(i)] + ['-'] + ['?']
            if k - i - 1 == 0:
                target = ['e'] + [str(k - i - 1)]
            else:
                target = ['e'] + ['-'] + [str(k - i - 1)]
            self.source.append(source)
            self.target.append(target)
            
    def carry_add_result(self,my_list,num = 0):
        total = sum(my_list) + len(my_list) - 1
        source = []
        target = []

        for i in range(total):
            source += [base_data[i + num]]

        source = ['|'] + source
        Sum = 0
        # 1 2  - 2 5  
        for i in range(len(my_list) - 1):
            Sum += my_list[i] + 1
            source[Sum] = '|'

        temp = copy.copy(source)
        
        for i in range(total):
            while len(temp) >= 3:
                if temp[-2] != '|' :
                    target = ['+'] + [temp[-2]] + ['|'] + [temp[-1]] + target
                    temp = temp[:-3]
                else:
                    target = ['|'] + [temp[-1]] + target
                    temp = temp[:-2]
        target = temp + target

        if num != 0:
            source = ['s','-'] + source
            target = ['e','-','|'] + target[1:]
        else:
            source[0] = 's'
            target[0] = 'e'


        return source,target

    def carry_sub_result(self,my_list):
        total = sum(my_list) + len(my_list) - 1
        source = []
        target = []

        for i in range(total):
            source += [base_data[i]]
        source = ['|'] + source
        Sum = 0
        # 
        for i in range(len(my_list) - 1):
            Sum += my_list[i] + 1
            source[Sum] = '|'
            if my_list[i] == 2:
                source[Sum - 2] = '-' 
        if source[-2] != '|':
            source[-2] = '-'

        temp = copy.copy(source)
        if my_list[0] == 2:
            for i in range(total):
                judge = True
                while len(temp) >= 2:
                    if temp[-2] == '|' and judge:
                        target = ['|'] +  [temp[-1]] + ['-','*'] + target
                        temp = temp[:-2]
                        judge = False
                    elif temp[-2] != '|' and judge:
                        target = ['|'] + ['0',temp[-2],temp[-1]] + target
                        temp = temp[:-3]
                        judge = True
                    elif temp[-2] == '|':
                        target = ['|'] +  [temp[-1]] + ['-','?'] + target
                        temp = temp[:-2]
                        judge = False
                    elif temp[-2] != '|' :
                        target = ['|'] +['1','-']+ [temp[-1]] + target
                        temp = temp[:-3]
                        judge = True

            target = temp + target
            source[0] = 's'
            target[0] = 'e'
        else:
            for i in range(total):
                judge = True
                while len(temp) >= 2:
                    if temp[-2] != '|' and judge:
                        target = ['|'] + ['*'] + ['-'] + [temp[-1]] + target
                        temp = temp[:-3]
                        judge = False
                    elif temp[-2] == '|' and judge:
                        target = ['|'] + [temp[-1]] + target
                        temp = temp[:-2]
                        judge = True
                    elif temp[-2] != '|':
                        target = ['|'] + ['?'] + ['-'] + [temp[-1]] + target
                        temp = temp[:-3]
                        judge = False
                    elif temp[-2] == '|' :
                        target = ['|'] + [temp[-1]] +['-','1']+ target
                        temp = temp[:-2]
                        judge = True

            target = temp + target
            source[0] = 's'
            target[0] = 'e'

        return source,target

    def carry(self):
        """
        Generates data involving carry-over and borrow rules for arithmetic operations.
        """
        test = DP(self.length)
        vectors = test.result()

        for vector in vectors:
            source,target = self.carry_add_result(vector)
            self.source.append(source)
            self.target.append(target)
            source,target = self.carry_add_result(vector,2)
            self.source.append(source)
            self.target.append(target)
            source,target = self.carry_sub_result(vector)
            self.source.append(source)
            self.target.append(target)



    def compute_mid_result(self,my_list):
        my_list = [3 if x == 2 else x for x in my_list]
        total = sum(my_list) + len(my_list) - 1
        source = []
        target = []

        for i in range(total):
            source += [base_data[i]]
        source = ['&'] + source
        Sum = 0
        for i in range(len(my_list) - 1):
            Sum += my_list[i] + 1
            source[Sum] = '&'

        source[0] = 's'

        return source

    def compute(self):
        """
        Prepares intermediate results for calculations involving carry-over and
        generates appropriate source-target pairs.
        """
        test = DP(self.length + 1)
        vectors = test.result()
        self.source.append(['s','A'])
        self.target.append(['e'])
        self.source.append(['s','A','B','C'])
        self.target.append(['e'])
        for vector in vectors:
            source = self.compute_mid_result(vector)
            target = []
            if source[-2] == '&':
                target = source[:-2]
            else:
                target = source[:-4]
            target[0] = 'e'
            self.source.append(source)
            self.target.append(target)

    def same_length(self):
        """
        Ensures all sequences in source and target have the same length by padding.
        """
        length = 0
        for i in range(len(self.source)):
            max_value = max(len(self.source[i]),len(self.target[i]))
            if length < max_value:
                length = max_value

        for i in range(len(self.source)):
            self.source[i] += [' '] * (length - len(self.source[i]))
            self.target[i] += [' '] * (length - len(self.target[i]))
        return length
    
            
    def generator(self):
        """
        Orchestrates the generation of source and target pairs by executing
        various stages of rule application and alignment.
        
        Returns:
            tuple: Aligned source and target sequences.
        """

        self.align()
        self.add()
        self.carry()
        self.compute()
        self.same_length()
        
        return self.source,self.target


