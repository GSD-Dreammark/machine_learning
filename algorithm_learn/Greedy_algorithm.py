# 贪心算法
# 即将问题拆分为小问题再求局部最优解
# eg:
# 给定一个非负整数数组，您最初定位在数组的第一个索引处。
# 数组中的每个元素表示该位置的最大跳转长度。
# 您的目标是以最小跳跃次数到达最后一个索引。
# 例：
# 输入： [2,3,1,1,4]
#  输出： 2
#  说明：到达最后一个索引的最小跳转次数为2。
#     从索引0跳转1步到1，然后从最后一个索引跳3步。
# 注意：
# 您可以假设您始终可以访问最后一个索引。
# arr=[2,1,1,1,4]
# step=0
# flag=True
# xb=0
# while len(arr)>1:
#     first=arr[0]
#     step += 1
#     last=0
#     for j in range(1,first+1):
#         if last>len(arr):
#             flag=False
#             break
#         if last<=arr[j]:
#             last=arr[j]
#             xb=j
#     if flag==False:
#         break
#     for i in range(xb):
#         arr.pop(0)


# **********************************
# 假设您有一个数组，其中第i 个元素是第i天给定股票的价格。
# 如果您只被允许完成最多一笔交易（即买入并卖出一股股票），请设计算法以找出最大利润。
# 请注意，在购买之前不能出售股票。
# 例1：
# 输入： [7,1,5,3,6,4]
#  输出： 5
#  说明：在第2天买入（价格= 1）并在第5天卖出（价格= 6），利润= 6-1 = 5。
#              不是7-1 = 6，因为售价需要大于购买价格。
# 例2：
# 输入： [7,6,4,3,1]
#  输出： 0
#  说明：在这种情况下，没有进行任何交易，即最大利润= 0。
arr1=[7,1,5,3,6,4]
money=0
while len(arr1)>=1:
    M_min=max(arr1)+1
    M_max=-1
    M_min_index = -1
    M_max_index = -1
    def getMaxMin(arr1,M_min,M_max,M_min_index,M_max_index):
        for i in range(len(arr1)):
            if arr1[i]<0:
                continue
            if M_min>=arr1[i]:
                M_min=arr1[i]
                M_min_index=i
            if M_max<=arr1[i]:
                M_max=arr1[i]
                M_max_index=i
        return M_min,M_min_index,M_max,M_max_index
    M_min,M_min_index,M_max,M_max_index=getMaxMin(arr1,M_min,M_max,M_min_index,M_max_index)
    print( M_min,M_min_index,M_max,M_max_index)
    if M_max_index<M_min_index:
        for i in range(M_min_index):
            arr1.pop(0)
    else:
        if money<M_max-M_min:
            money=M_max-M_min
        arr1.pop(0)
print(money)


