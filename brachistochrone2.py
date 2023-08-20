# 유전 알고리즘을 이용한 최속강하곡선 그리기
# 코드의 작성방식은 PEP8 (Program Enhance Proposal 8)에 의거합니다.

import copy
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from math import sqrt, cos, sin, tan, atan, pi, exp

r = 100                         # 맵 크기
x_len = 2*r                     # 그래프상 x범위
x_size = 100                    # 구간 개수(염기서열 길이)
y_len = 2*r                     # 그래프상 y범위
y_size = 2*r                    # 구간 개수

x = []
for i in range(x_size + 1):
    x.append(i*pi*r/x_size)

G = 9.80665                     # 중력가속도
T = sqrt(r / G)*pi              # 최속강하시간

# 정규화 함수: 배열의 msum과 ysize가 같도록 만듬
def generalization(m_arr0):
    # m_arr0 : 기울기 배열 입력
    m_arr = copy.deepcopy(m_arr0)   # 깊은 복사, 이래야 원래 값에 영향 x
    msum = 0                    # 기울기 합
    for i in m_arr:
        msum += i

    delta = y_len / msum * -1

    for i in range(len(m_arr)): 
        m_arr[i] *= delta
    return m_arr

# 시간 합 함수: 시간 합을 구함
def sum_time(m_arr):
    # m_arr : 기울기 배열 입력
    sum = 0
    v0 = 0
    for m in m_arr:
        M = abs(m)
        batch = r * pi / x_size
        h = M * batch
        # feat. 가속도 정의, m <= 0이라는 가정 하에 만듦
        # a = h * G / sqrt(batch**2 + h**2)
        a = G * M / sqrt(1+M**2)
        # 나중속도 산출
        if M == 0:
            after_v = v0
        else:
            # feat. 역학적E보존
            after_v = sqrt(2*G*h + v0**2)
        # feat. 가속도 정의 이용, t 산출
        if a == 0:
            if v0 == 0:
                return 1000
            else :
                t = pi/v0
        else:
            t = (after_v - v0)/a
        sum += t
        v0 = after_v
    return sum

# 적합도 함수: 룰렛 휠 방식, 선택압 상수 이용, 최대 적합도가 최소 적합도의 k배가 되도록 만듬
def fitness(t):
    # t : sum_time입력받음
    # 별로 차이가 나지 않아 제곱을 해 주는 것이 좋을 것 같음
    # f = (T/t)**2 * 100
    f = T/t*100
    return f

# 사이클로이드 그림 함수: 사이클로이드를 그림
def draw_cycloid(r):
  # r : 맵 크기 입력
  x = []                        # x좌표 리스트 만듦
  y = []                        # y좌표 리스트 만듦

  for theta in np.linspace(0, np.pi, 100): # theta변수를 -2π 에서 2π 까지 반복함
    x.append(r*(theta - sin(theta))) # x 리스트에 매개변수함수값을 추가시킴
    y.append(10 -(r*(1 - cos(theta)))) # y 리스트에 매개변수함수값을 추가시킴

  #plt.figure(figsize=(pi*2, 4))# 그래프 비율 조정
  plt.plot(x,y)                 # matplotlib.piplot을 이용해 그래프 그리기
  plt.title('Cycloid')
  plt.xlabel('x')
  plt.ylabel('y')
  #plt.xlim([0, r*pi])
  #plt.ylim([0, 2*r])
  plt.show()                    # 그래프 출력하기

# 유전 최속강하곡선 그림 함수: 유전 알고리즘으로 만들어진 사이클로이드를 그리는 함수
def draw_GA_Brachistochrone(m_arr, generation, fitness):
    # m_arr : 기울기 배열 입력
    # generation : 세대 입력
    # fitness : 적합도 입력
    y = [y_len]
    for i in range(x_size):
        y.append(y[-1] + m_arr[i])
    
    plt.plot(x, y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(f'GA_Brachistochrone_{generation}')
    plt.xlim([0, r*pi])
    plt.ylim([0, 2*r])
    plt.text(0.5, 0.05, f'Generation: {generation}\nFitness: {fitness:.2f}', 
             transform=plt.gca().transAxes, fontsize=10, ha='center')
    plt.show()

# 유전 함수: 유전하는 함수
def selection(chilren_num, min_m, max_gen, mut_chance, mut_rate, selection_rate):
    # 1. 초기화: 0세대 무작위 생성
    #초기 자녀 배열 생성, 자녀 수 num, 자녀 1명당 파라미터 수 x_size
    children = []               # 자녀 행렬
    parents = []                # 부모 행렬
    time_table = []             # 자녀 강하시간 배열
    fitness_table = []          # 자녀 적합도 배열: 부모 수만큼 생성
    gen = 0                     # 세대수
    generation = [0]            # 개체군(세대수) 행렬
    gen_maxfitness = []         # 개체군(세대 최고 적합도) 행렬
    gen_avefitness = []         # 개체군(세대 평균 적합도) 행렬
    gen_minfitness = []         # 개체군(세대 최저 적합도) 행렬
    sum_fitness = 0             # 세대 적합도 합
    max_index = 0               # 개체군 최대 적합도

    # children = [[자녀 1의 적합도, 자녀 1의 강하시간, 자녀 1의 염기서열], [자녀 2의 적합도, 자녀 2의 강하시간, 자녀 2의 염기서열], ...]
    # 자녀 행렬 생성
    for i in range(chilren_num):
        children.append([0, 0, []])
        
        # 자녀 염기서열 생성
        for j in range(x_size):
            randM = uniform(min_m, 0)
            children[i][2].append(randM)
        
        # 자녀 염기서열 정규화
        children[i][2] = generalization(children[i][2])
    
    # 2. 적합도 평가: 각 세대의 적합도 평가
    for i in range(chilren_num):
        time_table.append(sum_time(children[i][2]))
        children[i][1] = copy.deepcopy(time_table[i])

        fitness_table.append(fitness(time_table[-1]))
        children[i][0] = copy.deepcopy(fitness_table[i])
        sum_fitness += fitness_table[i]

    # 0세대 적합도 내림차순으로 정렬
    children.sort()             # 먼저 적합도 정렬, 이후 강하시간 정렬, 이후 염기서열 정렬, 그러나 적합도 정렬에서 끝.
    fitness_table.sort()        # 자녀 강하시간 
    time_table.sort()

    gen_maxfitness.append(fitness_table[0])
    gen_avefitness.append(sum_fitness/chilren_num)
    gen_minfitness.append(fitness_table[-1])

    # 유전 시작 =====================================================================
    
    while gen < max_gen:
        # 세대 카운팅
        gen += 1
        generation.append(gen)
        parents = []            # 부모
        inherit = []            # 상위 selection rat염기서열 저장 공간
        # 0. 아이 유전자를 다음 세대로 전달
        # children.sort()
        parents = copy.deepcopy(children)
        children = []              # 아이
        # 1. 상위 selection rate만 다음 세대로 전달
        selection_chilren_num = round(chilren_num*selection_rate)
        for i in range(selection_chilren_num):
            inherit.append(parents[i])
        children = copy.deepcopy(inherit)
        
        # 2. 50% 교차시킴
        #임시 저장 변수
        val = 0
        for i in range(chilren_num//4):             #4의 배수로 num 입력해야 함
            # 교차 = [교차될 사람 1, 교차될 사람 2, 교차될 포인트 1, 교차될 포인트 2]
            cross_arr = []
            for j in range(4):
                if j < 2:
                    cross_arr.append(int(uniform(0, selection_chilren_num)))
                else:
                    cross_arr.append((int(uniform(0, x_size))))
                
                if j % 2 == 1:
                    while cross_arr[j-1] == cross_arr[j]:
                        if j == 1:
                            cross_arr[j] = int(uniform(0, selection_chilren_num))
                            cross_arr[j-1] = int(uniform(0, selection_chilren_num))
                        else:
                            cross_arr[j] = int(uniform(0, x_size))
                            cross_arr[j-1] = int(uniform(0, x_size))
                    
                    #0 1, 2 3은 각각 오름차순으로 정렬
                    if cross_arr[j-1] > cross_arr[j]:
                        cross_arr[j-1], cross_arr[j] = cross_arr[j], cross_arr[j-1]

            #children에 변화시염기서열 2개씩 추가
            children.append(inherit[cross_arr[0]])
            children.append(inherit[cross_arr[1]])
            #j번째 인덱스를 swap
            #2점교차
            for j in range(cross_arr[2], cross_arr[3]+1):
                children[-1][2][j], children[-2][2][j] = children[-2][2][j], children[-1][2][j]
        
        # 3. 50% 돌연변이
        inherit *= 3
        for i in inherit:
            children.append(i)
        
        for i in range(chilren_num//10*4, chilren_num//10*9):
            for j in range(x_size):
                point = uniform(0, 1)
                if(point < mut_chance):
                    rate = uniform(1-mut_rate, 1+mut_rate)
                    children[i][2][j] *= rate
        # 4. 무작위 10% 생성
        for i in range(chilren_num//10):
            children.append([])
            children[-1].append(0)
            children[-1].append(0)
            bucket = []
            b_fitness = 0
            #적합도 90 이상인 아이들만 유전시킴
            #while b_fitness < 90:
            for j in range(x_size):
                theta = uniform(atan(min_m), 0)   #-2 이상 0 미만 실수값 입력, 단위 라디안
                #10개 숫자 평균적 합이 10이 되도록 맞춰줌 
                randM = tan(theta)
                #randM = rand(-2, 0)        # 기울기 자체를 랜덤값으로 결정
                bucket.append(randM)
                #b_fitness = fitness(sum_time(bucket))

            children[-1].append(bucket)
        # 5. 정규화
        for i in range(chilren_num):
            children[i][2] = generalization(children[i][2])
        
        #gen 출력
        if gen % 100 == 0:
            print('gen :', gen)
            #draw_GA_Brachistochrone(children[0][2], gen, children[0][0])

    
        # 6. 적합도 계산
        fitness_table = []
        time_table = []
        sum_fitness = 0
        for i in range(chilren_num):
            time_table.append(sum_time(children[i][2]))
            children[i][1] = copy.deepcopy(time_table[i])

            fitness_table.append(fitness(time_table[-1]))
            children[i][0] = copy.deepcopy(fitness_table[i])
            sum_fitness += fitness_table[i]

        fitness_table.sort()
        time_table.sort()
        children.sort()

        gen_maxfitness.append(fitness_table[0])
        if gen_maxfitness[-1] > gen_maxfitness[-2]:
            max_index = gen
        gen_avefitness.append(sum_fitness/chilren_num)
        gen_minfitness.append(fitness_table[-1])
        """
        plt.plot(list(range(children_num)), fitness_table)
        plt.title('Fitness Table')
        plt.xlim([0, children_num])
        plt.ylim([0, 100])
        plt.show()
        """
        

    # 7. 적합도 그래프 출력
    plt.figure(figsize = (100, 100))
    plt.plot(generation, gen_maxfitness, marker = 'o')
    plt.plot(generation, gen_avefitness, marker = 'o')
    plt.plot(generation, gen_minfitness, marker = 'o')
    plt.title('Fitnesses of All Generations')
    plt.legend(['max', 'ave', 'min'])
    #plt.legend(['max', 'ave'])
    plt.show()
    #print('max', gen_maxfitness)
    #print('ave', gen_avefitness)
    #print('min', gen_minfitness)
    

    draw_GA_Brachistochrone(children[0][2], gen, children[0][0])
    #세대출력
    print('')
    print('gen :', max_index)
    print('max fitness :', children[0][0], '%')
    print('GA_Brachistochrone time :', children[0][1], 's')
    print('time difference between GA_B and C', children[0][1] - T, 's')
    print('')

"""알고리즘 개요"""
#1. 자식세대 유전자를 부모 세대로 전달(성숙)
#2. 적합도 상위 10% 자녀들만 다음 세대로 유전.
#3. children세대로 선택염기서열 전달
#4. 전체의 50% 교차됨
#5. 전체의 50% 돌연변이를 일으킴
#6. 10%는 그대로 전달, 10%는 무작위 생성됨
#6. 정규화
#7. 세대 종료, 다음 세대로 이동

max_gen = 300                     # 최대 세대 수
chlidren_num = 120              # 자녀 수 : 4와 10의 최소공배수 입력
min_m = -10                     # 최소 기울기 값
mut_chance = 0.3                # 돌연변이 비율
mut_rate = 0.3                  # 돌연변이시 바꾸는 최대 비율 : 양수 입력
selection_rate = 0.1            # 선택 비율

print('')
print('Finding Brachistochrone by Genetic Algorithm')
print('Cycloid time :', T, 's')
print('')
# 자녀 정보 출력
print('total children :', chlidren_num)
print('DNA sequence length : ', x_size)
print('min_m :', min_m)
print('mut_chance :', mut_chance)
print('mut_rate :', mut_rate)
print('batch :', x_size)
print('r :', r)
print('')

# 유전 알고리즘 시작
selection(chlidren_num, min_m, max_gen, mut_chance, mut_rate, selection_rate)
draw_cycloid(r)