# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import re
import copy
import os
import string
import xlrd
from Utils.get_tokens import *
import pickle
from difflib import SequenceMatcher


keywords_0 = ('auto', 'typedf', 'const', 'extern', 'register', 'static', 'volatile', 'continue', 'break',
              'default', 'return', 'goto', 'else', 'case')

keywords_1 = ('catch', 'sizeof', 'if', 'switch', 'while', 'for')

keywords_2 = ('memcpy', 'wmemcpy', '_memccpy', 'memmove', 'wmemmove', 'memset', 'wmemset', 'memcmp', 'wmemcmp', 'memchr',
              'wmemchr', 'strncpy', 'lstrcpyn', 'wcsncpy', 'strncat', 'bcopy', 'cin', 'strcpy', 'lstrcpy', 'wcscpy', '_tcscpy',
              '_mbscpy', 'CopyMemory', 'strcat', 'lstrcat', 'fgets', 'main', '_main', '_tmain', 'Winmain', 'AfxWinMain', 'getchar',
              'getc', 'getch', 'getche', 'kbhit', 'stdin', 'm_lpCmdLine', 'getdlgtext', 'getpass', 'istream.get', 'istream.getline',
              'istream.peek', 'istream.putback', 'streambuf.sbumpc', 'streambuf.sgetc', 'streambuf.sgetn', 'streambuf.snextc', 'streambuf.sputbackc',
              'SendMessage', 'SendMessageCallback', 'SendNotifyMessage', 'PostMessage', 'PostThreadMessage', 'recv', 'recvfrom', 'Receive',
              'ReceiveFrom', 'ReceiveFromEx', 'CEdit.GetLine', 'CHtmlEditCtrl.GetDHtmlDocument', 'CListBox.GetText', 'CListCtrl.GetItemText',
              'CRichEditCtrl.GetLine', 'GetDlgItemText', 'CCheckListBox.GetCheck', 'DISP_FUNCTION', 'DISP_PROPERTY_EX', 'getenv', 'getenv_s', '_wgetenv',
              '_wgetenv_s', 'snprintf', 'vsnprintf', 'scanf', 'sscanf', 'catgets', 'gets', 'fscanf', 'vscanf', 'vfscanf', 'printf', 'vprintf', 'CString.Format',
              'CString.FormatV', 'CString.FormatMessage', 'CStringT.Format', 'CStringT.FormatV', 'CStringT.FormatMessage', 'CStringT.FormatMessageV',
              'vsprintf', 'asprintf', 'vasprintf', 'fprintf', 'sprintf', 'syslog', 'swscanf', 'sscanf_s', 'swscanf_s', 'swprintf', 'malloc',
              'readlink', 'lstrlen', 'strchr', 'strcmp', 'strcoll', 'strcspn', 'strerror', 'strlen', 'strpbrk', 'strrchr', 'strspn', 'strstr',
              'strtok', 'strxfrm', 'kfree', '_alloca')

keywords_3 = ('_strncpy*', '_tcsncpy*', '_mbsnbcpy*', '_wcsncpy*', '_strncat*', '_mbsncat*', 'wcsncat*', 'CEdit.Get*', 'CRichEditCtrl.Get*',
              'CComboBox.Get*', 'GetWindowText*', 'istream.read*', 'Socket.Receive*', 'DDX_*', '_snprintf*', '_snwprintf*')

keywords_5 = ('*malloc',)

try:
    xread = xlrd.open_workbook(r'../Utils/function.xls')
except Exception as e:
    xread = xlrd.open_workbook(r'../../Utils/function.xls')
keywords_4 = []
for sheet in xread.sheets():
    col = sheet.col_values(0)[1:]
    keywords_4 += col
#print keywords_4

typewords_0 = ('short', 'int', 'long', 'float', 'doubule', 'char', 'unsigned', 'signed', 'void' ,'wchar_t', 'size_t', 'bool')
typewords_1 = ('struct', 'union', 'enum')
typewords_2 = ('new', 'delete')
operators = ('+', '-', '*', '/', '=', '%', '?', ':', '!=', '==', '<<', '&&', '||', '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|=', '>>=', '<<=', '++', '--', '>>', '<=', '>=', ',', '^', '?:', '.*', '->*')
function = '^[_a-zA-Z][_a-zA-Z0-9]*$'
variable = '^[_a-zA-Z][_a-zA-Z0-9(->)?(\.)?]*$'
number = '[0-9]+'
stringConst = '(^\'[\s|\S]*\'$)|(^"[\s|\S]*"$)'
constValue = ['NULL', 'false', 'true']
phla = '[^a-zA-Z0-9_]'
space = '\s'
spa = ''


def isinKeyword_3(token):
    for key in keywords_3:
        if len(token) < len(key)-1:
            return False
        if key[:-1] == token[:len(key)-1]:
            return True
        else:
            return False


def isinKeyword_5(token):
    for key in keywords_5:
        if len(token) < len(key)-1:
            return False
        if token.find(key[1:]) != -1:
            if "_" in token:
                return False
            else:
                return True
        else:
            return False


def isphor(s, liter):
    m = re.search(liter, s)
    if m is not None:
        return True
    else:
        return False

def var(s):
    m = re.match(function, s)
    if m is not None:
        return True
    else:
        return False

def CreateVariable(string, token):
    length = len(string)
    stack1 = []
    s = ''
    i = 0
    while (i < length):
        if var(string[i]):  
            #if i + 1 < length and (string[i + 1] == '->' or string[i + 1] == '.'): 
            #    stack1.append(string[i])
            #    stack1.append(string[i + 1])
            #    i = i + 2

            #else:
            while stack1 != []:
                s = stack1.pop() + s
            s = s + string[i]
            token.append(s)
            s = ''
            i = i + 1
        else:
            token.append(string[i])
            i = i + 1

def is_type_token(token):
    """토큰이 C++ 타입인지 확인"""
    # 기본 타입들
    basic_types = {
        'void', 'int', 'char', 'float', 'double', 'bool', 
        'short', 'long', 'signed', 'unsigned',
        'size_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
        'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
    }
    
    # 사용자 정의 타입 패턴 (대문자로 시작하는 경우)
    if token and token[0].isupper():
        return True
    
    return token in basic_types

def is_class_or_namespace(token, _variable_dict, _func_dict):
    """토큰이 클래스나 네임스페이스인지 확인"""
    # 이미 함수나 변수로 매핑된 토큰은 제외
    if token in _variable_dict or token in _func_dict:
        return False
    
    # 대문자로 시작하는 토큰은 클래스/네임스페이스로 간주
    if token and token[0].isupper():
        return True
    
    # STL이나 일반적인 클래스 이름 패턴
    common_classes = {
        'string', 'vector', 'list', 'map', 'set', 'unordered_map',
        'unique_ptr', 'shared_ptr', 'weak_ptr', 'Document', 'Frame'
    }
    
    return token in common_classes

def mapping(list_sentence):
    list_code = []
    list_func = []
    for code in list_sentence:
        #print code
        _string = ''
        for c in code:
            _string = _string + ' ' + c
        _string = _string[1:]
        list_code.append(_string)
    
    #print list_code    
    _func_dict = {}
    _variable_dict = {}
    index = 0
    while index < len(list_code):
        string = []
        token = []
        j = 0
        str1 = copy.copy(list_code[index])
        i = 0
        tag = 0
        strtemp = ''
        while i < len(str1):
            if tag == 0:
                if isphor(str1[i], space):  
                    if i > 0:
                        string.append(str1[j:i])
                        j = i + 1

                    else:
                        j = i + 1
                    i = i + 1

                elif i + 1 == len(str1):
                    string.append(str1[j:i + 1])
                    break

                elif isphor(str1[i], phla):  
                    if i + 1 < len(str1) and str1[i] == '-' and str1[i + 1] == '>':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '<' and str1[i + 1] == '<':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '>' and str1[i + 1] == '>':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '&' and str1[i + 1] == '&':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '|' and str1[i + 1] == '|':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '|' and str1[i + 1] == '=':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '=' and str1[i + 1] == '=':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '!' and str1[i + 1] == '=':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '+' and str1[i + 1] == '+':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '-' and str1[i + 1] == '-':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '+' and str1[i + 1] == '=':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif i + 1 < len(str1) and str1[i] == '-' and str1[i + 1] == '=':
                        string.append(str1[i] + str1[i + 1])
                        j = i + 2
                        i = i + 2

                    elif str1[i] == '"':  
                        strtemp = strtemp + str1[i]
                        i = i + 1
                        tag = 1

                    elif str1[i] == '\'':  
                        strtemp = strtemp + str1[i]
                        i = i + 1
                        tag = 2

                    else:
                        string.append(str1[i])
                        j = i + 1
                        i += 1

                else:
                    i += 1
            elif tag == 1:
                if str1[i] != '"':
                    strtemp = strtemp + str1[i]
                    i = i + 1

                else:
                    strtemp = strtemp + str1[i]
                    string.append(strtemp)
                    strtemp = ''
                    tag = 0
                    j = i + 1
                    i += 1

            elif tag == 2:
                if str1[i] != '\'':
                    strtemp = strtemp + str1[i]
                    i = i + 1

                else:
                    strtemp = strtemp + str1[i]
                    string.append(strtemp)
                    strtemp = ''
                    tag = 0
                    j = i + 1
                    i += 1

        count = 0
        for sub in string:
            if sub == spa:
                count += 1

        for i in range(count):
            string.remove('')

        CreateVariable(string, token)

        j = 0
        while j < len(token):
            if token[j] in constValue:
                token[j] = token[j]
                j += 1

            elif j < len(token) and isphor(token[j], variable): 
                if j + 1 < len(token) and token[j] == 'const' and is_class_or_namespace(token[j + 1], _variable_dict, _func_dict):
                    j = j + 2
                elif j + 2 < len(token) and token[j] == 'struct' and (token[j + 2] == '*' or token[j + 2] == '{'):
                    j = j + 3
                elif (token[j] in keywords_0) or (token[j] in typewords_0) or (token[j] in typewords_1 or token[j] in typewords_2): 
                    j = j + 1
                elif j + 2 < len(token) and token[j + 1] == ':' and token[j + 2] == ':':
                    j = j + 3
                elif j - 2 >= 0 and token[j - 1] == ':' and token[j - 2] == ':':
                    j = j + 1
                elif j - 1 >= 0 and j + 1 < len(token) and token[j-1] == 'new' and token[j + 1] == '[':
                    j = j + 2
                elif j - 1 >= 0 and token[j - 1] == '}' and is_class_or_namespace(token[j], _variable_dict, _func_dict):
                    j = j + 1
                elif j + 1 < len(token) and token[j + 1] == '(': 
                    #print(token[j])
                    if token[j] in keywords_1: 
                        j = j + 2

                    elif token[j] in keywords_2: 
                        #print('3', token[j])
                        j = j + 2

                    elif isinKeyword_3(token[j]): 
                        #print('4', token[j])
                        j = j + 2

                    elif token[j] in keywords_4: 
                        #print('5', token[j])
                        j = j + 2

                    elif isinKeyword_5(token[j]): 
                        #print('6', token[j])
                        j = j + 2

                    else:
                        #print('7',token[j])
                        if "good" in token[j] or "bad" in token[j]:
                            list_func.append(str(token[j]))
                        if token[j] in _func_dict.keys():
                            token[j] = _func_dict[token[j]]
                        else:
                            list_values = _func_dict.values()
                            if len(list_values) == 0:
                                _func_dict[token[j]] = 'func_0'
                                token[j] = _func_dict[token[j]]
                                
                            else:
                                if token[j] in _func_dict.keys():
                                    token[j] = _func_dict[token[j]]
                                else:
                                    list_num = []
                                    for value in list_values:
                                        list_num.append(int(value.split('_')[-1]))

                                    _max = max(list_num)
                                    _func_dict[token[j]] = 'func_' + str(_max+1)
                                    token[j] = _func_dict[token[j]]
                            j = j + 2

                elif j + 1 < len(token) and (not isphor(token[j + 1], variable)):
                    if token[j + 1] == '*':
                        # 2. 타입 다음에 오는 * 보호 (포인터 타입 선언)
                        if is_type_token(token[j]):
                            j = j + 2
                        # 3. 클래스/네임스페이스 다음에 오는 * 보호
                        elif is_class_or_namespace(token[j], _variable_dict, _func_dict):
                            j = j + 2
                        elif j + 2 < len(token) and token[j + 2] == 'const':
                            j = j + 3

                        elif j - 1 >= 0 and token[j - 1] == 'const':
                            j = j + 2

                        elif j - 1 > 0 and (token[j - 1] in operators): 
                            list_values = _variable_dict.values()
                            if len(list_values) == 0:
                                _variable_dict[token[j]] = 'variable_0'
                                token[j] = _variable_dict[token[j]]
                                
                            else:
                                if token[j] in _variable_dict.keys():
                                    token[j] = _variable_dict[token[j]]
                                else:
                                    list_num = []
                                    for value in list_values:
                                        list_num.append(int(value.split('_')[-1]))

                                    _max = max(list_num)
                                    _variable_dict[token[j]] = 'variable_' + str(_max+1)
                                    token[j] = _variable_dict[token[j]]
                            j = j + 2

                        elif j + 2 < len(token) and token[j + 2] == ')':
                            j = j + 2

                        elif j - 2 > 0 and (token[j - 1] == '(' and token[j - 2] in operators):  
                            list_values = _variable_dict.values()
                            if len(list_values) == 0:
                                _variable_dict[token[j]] = 'variable_0'
                                token[j] = _variable_dict[token[j]]
                                
                            else:
                                if token[j] in _variable_dict.keys():
                                    token[j] = _variable_dict[token[j]]
                                else:
                                    list_num = []
                                    for value in list_values:
                                        list_num.append(int(value.split('_')[-1]))

                                    _max = max(list_num)
                                    _variable_dict[token[j]] = 'variable_' + str(_max+1)
                                    token[j] = _variable_dict[token[j]]
                            j = j + 2


                        else:
                            list_values = _variable_dict.values()
                            if len(list_values) == 0:
                                _variable_dict[token[j]] = 'variable_0'
                                token[j] = _variable_dict[token[j]]

                            else:
                                if token[j] in _variable_dict.keys():
                                    token[j] = _variable_dict[token[j]]
                                else:
                                    list_num = []
                                    for value in list_values:
                                        list_num.append(int(value.split('_')[-1]))

                                    _max = max(list_num)
                                    _variable_dict[token[j]] = 'variable_' + str(_max+1)
                                    token[j] = _variable_dict[token[j]]

                            j = j + 2

                    else:
                        list_values = _variable_dict.values()
                        if len(list_values) == 0:
                            _variable_dict[token[j]] = 'variable_0'
                            token[j] = _variable_dict[token[j]]
                                
                        else:
                            if token[j] in _variable_dict.keys():
                                token[j] = _variable_dict[token[j]]
                            else:
                                list_num = []
                                for value in list_values:
                                    list_num.append(int(value.split('_')[-1]))

                                _max = max(list_num)
                                _variable_dict[token[j]] = 'variable_' + str(_max+1)
                                token[j] = _variable_dict[token[j]]
                        j = j + 2

                elif j + 1 == len(token):
                    list_values = _variable_dict.values()
                    if len(list_values) == 0:
                        _variable_dict[token[j]] = 'variable_0'
                        token[j] = _variable_dict[token[j]]
                                
                    else:
                        if token[j] in _variable_dict.keys():
                            token[j] = _variable_dict[token[j]]
                        else:
                            list_num = []
                            for value in list_values:
                                list_num.append(int(value.split('_')[-1]))

                            _max = max(list_num)
                            _variable_dict[token[j]] = 'variable_' + str(_max+1)
                            token[j] = _variable_dict[token[j]]
                        break

                else:
                    j += 1

            elif j < len(token) and isphor(token[j], number): 
                j += 1

            elif j < len(token) and isphor(token[j], stringConst): 
                j += 1

            else:
                j += 1

        stemp = ''
        i = 0
        while i < len(token):
            if i == len(token) - 1:
                stemp = stemp + token[i]
            else:
                stemp = stemp + token[i] + ' '
            i += 1

        list_code[index] = stemp
        index += 1

    #print list_code
    #print _variable_dict
    return list_code, len(_variable_dict)


def extract_variable_tokens(original_code):
    """
    원본 코드와 매핑된 코드를 비교하여 variable_n에 대응하는 원본 변수명 찾기
    
    Args:
        original_code: 원본 코드 문자열
        mapped_code: variable_n 형식으로 매핑된 코드 문자열
    
    Returns:
        list: variable_n에 대응하는 원본 변수명 리스트
    """
    # /* ... */ 주석 제거
    del_ano_code = re.sub(r'/\*.*?\*/', '', original_code, flags=re.DOTALL)
    # // 주석 제거
    del_ano_code = re.sub(r'//.*', '', del_ano_code)

    original_lines = del_ano_code.split('\n')
    instatements = []
    for line in original_lines:
        line_tokens = create_tokens(line)
        instatements.append(line_tokens)

    mapped_lines,_ = mapping(instatements)

    # 변수 매핑 딕셔너리 (variable_n -> original_name)
    variable_mapping = {}
    
    # 각 라인 쌍을 비교
    for orig_line, map_line in zip(original_lines, mapped_lines):
        if orig_line.strip() == map_line.strip():
            continue  # 동일한 라인은 스킵
        
        # 매핑된 라인에서 variable_n 패턴 추출
        mapped_vars = re.findall(r'\b(variable_\d+)\b', map_line)
        
        # 두 라인의 토큰 시퀀스 비교
        orig_tokens = create_tokens(orig_line)
        map_tokens = create_tokens(map_line)
        
        # 시퀀스 매처를 사용하여 변경된 부분 찾기
        matcher = SequenceMatcher(None, orig_tokens, map_tokens)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # 교체된 부분 처리
                for i in range(i1, i2):
                    if i < len(orig_tokens) and isphor(orig_tokens[i], variable):
                        for j in range(j1, j2):
                            if j < len(map_tokens) and re.match(r'variable_\d+', map_tokens[j]):
                                variable_mapping[map_tokens[j]] = orig_tokens[i]
    
    # 키 순서대로 정렬하고 리스트로 변환
    original_variables = [variable_mapping[key] for key in sorted(variable_mapping.keys(), key=lambda x: int(x.split('_')[1]))]
    
    return original_variables


if __name__ == '__main__':
    #file_path=r'D:\Documents\研究课题\Attack\Program\中间数据备份\Sample\_pos_21453'
    file_path=r'D:\Study\Mainstudy\SVulAttack\SVulAttack\test.cpp'
    with open(file_path,'r') as f:
        lines = f.readlines()
    instatements=[]
    for line in lines:
        line_tokens = create_tokens(line)
        instatements.append(line_tokens)
    list_code,_ = mapping(instatements)
    for line in list_code:
        print(line)
    # print(list_code)
