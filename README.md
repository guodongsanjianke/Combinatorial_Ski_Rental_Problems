# Combinatorial Ski Rental Problem: Robust and Learning-Augmented Algorithms

This repository contains the official Python implementation of the experiments and results presented in the NeurIPS 2025 paper:

> **[Combinatorial Ski Rental Problem: Robust and Learning-Augmented Algorithms](https://neurips.cc/virtual/2025/poster/117092)**

---



## Folder Structure and Main Files

```bash
├── LA-SOAC_main_paper/
│   ├── main_average_competitive.py          
│   ├── basic_func.py                      
│   ├── CSR.py                              
│   └── Consistency_Robustness/
│       ├── consistency_robustness.py
│		└── basic_func.py  
├── LA-SOAC_sup_materical/
│   ├── main_average_competitive_ratio.py  
│   ├── basic_func.py                      
│   ├── CSR.py                                                    
├── LA-SOAC_to_other_variants/
│   ├── consistency-robustness.py 
│	├── consistency-robustness_MMSR.py
│   ├── compare_to_MSSR.py
│   ├── basic_func.py                                                   
│   └── CSR.py                     
├── SOAC_main_paper/
│   ├── case_study_three_items.py  
│   ├── basic_func.py                      
│   ├── CSR.py                             
│   └── upgrading.py                      
├── SOAC_sup_materical/
│   ├── SOAC_multi_items.py
│   ├── basic_func.py                      
│   ├── CSR.py                              
│   └── upgrading.py                       
└── SOAC_to_other_variants/
    ├── SOAC_to_multi_shop_ski rental.py
    ├── SOAC_to_multi_slope_ski rental.py
    ├── basic_func.py                       
    ├── CSR.py                              
    └── upgrading.py                       
```



### 1. `LA-SOAC_main_paper`

- **Purpose**: Contains code for the LA-SOAC algorithm results presented in the main text of the paper.

- Main File: 

  ```
  main_average_competitive.py
  ```

  - Runs experiments to compute the average competitive ratio for the LA-SOAC algorithm.

- Supporting Files:

  - `basic_func.py`
  - `CSR.py`
  - `upgrading.py`

### 2. `LA-SOAC_main_paper/Consistency_Robustness`

- **Purpose**: Contains code for analyzing the trade-off between consistency and robustness in the LA-SOAC algorithm, as discussed in the main text.

- Main File: 

  ```
  consistency_robustness.py
  ```

  - Runs experiments to evaluate the consistency and robustness properties of the LA-SOAC algorithm.

### 3. `LA-SOAC_sup_materical`

- **Purpose**: Contains code for the LA-SOAC algorithm results presented in the supplementary material (appendix) of the paper.

- Main File: 

  ```
  main_average_competitive_ratio.py
  ```

  - Runs experiments to compute the average competitive ratio for the LA-SOAC algorithm in the appendix.

### 4. `LA-SOAC_to_other_variants`

- **Purpose**: Contains code for comparing the LA-SOAC algorithm with the LA-MSSR algorithm.

### 5. `SOAC_main_paper`

- **Purpose**: Contains code for the SOAC algorithm results presented in the main text of the paper.

### 6. `SOAC_sup_materical`

- **Purpose**: Contains code for the SOAC algorithm results presented in the supplementary material (appendix) of the paper.

### 7. `SOAC_to_other_variants`

- **Purpose**: Contains code for experiments where the SOAC algorithm is applied to solve other problem variants, as discussed in the appendix.

## Usage

1. **Navigate to the desired folder** (e.g., `LA-SOAC_main_paper`).

2. Run the main file using Python. For example:

   ```bash
   python main_average_competitive.py
   ```

3. Ensure supporting files (`basic_func.py`, `CSR.py`, `upgrading.py`) are in the same directory or accessible in the Python path.

## Notes

- The supporting files (`basic_func.py`, `CSR.py`, `upgrading.py`) are not standalone scripts but are imported by the main files.

## Required Packages

The following Python packages are required to run the code, based on the imported modules. Standard library modules (e.g., `copy`, `random`, `math`, `collections`, `os`, `argparse`, `concurrent.futures`) are included with Python and do not need installation.

### Third-Party Packages

- `numpy>=1.26.4`: For numerical computations.
- `tqdm>=4.66.5`: For progress bars in loops.
- `matplotlib>=3.9.2`: For plotting results.
- `pandas>=2.2.3`: For data manipulation and analysis.
- `openpyxl>=3.1.5`: For reading/writing Excel files.
- `torch>=2.4.1`: For GPU-accelerated computations using PyTorch.

These packages are equivalent to the content of a `requirements.txt` file, which would look like:

```
numpy>=1.26.4
tqdm>=4.66.5
matplotlib>=3.9.2
pandas>=2.2.3
openpyxl>=3.1.5
torch>=2.4.1
```



