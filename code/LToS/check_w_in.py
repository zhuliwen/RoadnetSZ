def check_nohup(filename, content, mode=''):
    NUM_NEXT_LINES = 0
    
    with open(filename, 'r', encoding='UTF-8') as file:
        num_rest_lines = 0
        last_lines = []
        for linea in file:
            linea=linea.replace("\n","")

            if linea.startswith(mode) and content in linea:
                for last_line in last_lines:
                    if len(last_line) > 50:
                        last_line = last_line[:50] + ' ...'
                    print(last_line)
                # print('--------> ', linea)
                end = -1
                while True:
                    pattern = "'time': "
                    beg = linea.find(pattern, end+1)
                    if beg == -1:
                        break
                    end = linea.find(',', beg+1)
                    print(linea[beg+len(pattern):end], end=":")

                    beg = linea.find(content, end+1)
                    end = linea.find(']', beg+1)
                    print(linea[beg+len(content):end])
                num_rest_lines = NUM_NEXT_LINES # print this line and next N line
            else:
                if num_rest_lines > 0:
                    num_rest_lines -= 1
                    if len(linea) > 50:
                        linea = linea[:50] + ' ...'
                    print(linea)
                    if num_rest_lines == 0:
                        print('----------END----------')

            last_lines.append(linea)
            if len(last_lines) == NUM_NEXT_LINES+1:
                last_lines = last_lines[1:]


# check_nohup(filename=r"nohup.out", content="Epoch")
# check_nohup(filename=r"nohup.out", content="intersection")
# check_nohup(filename=r"nohup.out", content="rror")
import sys
with open('w_in_test.txt', 'w') as f:
    sys.stdout = f
    check_nohup(filename=r"nohup.out", content="'w_in': [", mode='test: ')
# check_nohup(filename=r"nohup.out", content="succeed in reloading model for initialization from")
# check_nohup(filename=r"nohup.out", content="Error occurs when making rewards")
# check_nohup(filename=r"nohup.out", content="succeed in loading model")
# check_nohup(filename=r"nohup.out", content="value")
