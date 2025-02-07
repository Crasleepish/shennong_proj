from typing import List, Callable, Any

def process_in_batches(data_list: List[Any], 
                       func: Callable[[List[Any]], Any], 
                       batch_size: int = 1000) -> List[Any]:
    """
    将 data_list 按照 batch_size 分成若干子列表，并对每个子列表调用函数 func。
    
    :param data_list: 需要处理的原始列表
    :param func: 处理函数，接受一个列表作为参数，并返回处理结果
    :param batch_size: 每个子列表的最大长度，默认为 1000
    :return: 返回一个列表，包含对每个子列表调用 func 后的返回结果
    """
    results = []
    # 从索引 0 开始，每次切出 batch_size 长度的子列表
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        result = func(batch)
        results.append(result)
    return results