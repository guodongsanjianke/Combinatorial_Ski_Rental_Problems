from CSR import csr
from tqdm import trange
from basic_func import opt_line
import openpyxl

wb = openpyxl.Workbook()

n = 3
b = [594, 576, 560]
r = [1, 1.2, 1.3]

# Define the price of a combination of items
b_dict = {}
for idx in trange(1,n+1):
    b_dict[(idx,)] = b[idx-1]

# compute opt result
opt_lines = [opt_line(b=0.0,r=r[0])]
opt_lines.append(opt_line(b=b[n-1], r=0.0))

print("opt_lines", opt_lines)

fixed_partitions = [
    [tuple([i])] for i in range(1,n+1)
]

# compute cr

result = csr(b, r, b_dict=b_dict, opt_lines=opt_lines, fixed_partitions=fixed_partitions)
print(result['CR'])

# Non-upgrading, the probability of 2 layers

all_shop = result['all_shop']

sheet1 = wb.active
sheet1.title = "multi-shop"
sheet1.append(["Partitions of Items", "The probability of the first layer", "The probability of the second layer"])


for shop in all_shop:
    print(str(shop.items_split)+":")
    print(f'The probability of the first layer:{shop.prob}')
    sheet1.append([str(shop.items_split), shop.prob, str(shop.p_list)])



wb.save("multi-shop.xlsx")
print("Data has been successfully written to multi-shop.xlsx")

