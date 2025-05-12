import os

for sample in os.listdir('./dataset'):
    input_dir = f'./dataset/{sample}/inputs'
    output_dir = f'./dataset/{sample}/outputs'
    # inputs = os.listdir(input_dir)
    # for input_item in inputs:
    #     if input_item.endswith('.txt'):
    #         if input_item in ['questions.txt', 'captions.txt']:
    #             continue
    #         print(input_item, sample)
    #         with open(os.path.join(input_dir, input_item), 'r', encoding='utf-8') as f:
    #             input_data = f.read()
    outputs = os.listdir(output_dir)
    if any('object' in item for item in outputs):
        print(outputs, sample)
        # with open(os.path.join(output_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
        #     label_data = f.read()
        #     label_data = label_data.splitlines()
        #     print(label_data)

    # pass
