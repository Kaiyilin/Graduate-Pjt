def maximumProduct(nums):
    nums.sort(reverse=True)
    print(nums[-2])
    #print(nums_sorted)
    #max_output = nums_sorted[0] * nums_sorted[1] * nums_sorted[2]
    return max([nums[0] * nums[1] * nums[2], nums[0] * nums[-1] * nums[-2]])


print(maximumProduct([1,2,3,4,-5,-6]))