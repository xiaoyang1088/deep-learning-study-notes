#数据准备
#从完整类别表中固定选取类别，检查减压图片，并生成训练、验证和最终测试清单

import argparse
import csv
import hashlib
import json 
import random 
import re

from collections import Counter,defaultdict     #collections是Python提供的一组容器工具，Counter是计数器，defaultdict是带默认值的字典 
from io import BytesIO
from pathlib import Path,PurePosixPath  #Posix是采用/分隔的路径风格

from PIL import Image

#读取并检查完整的1000类列表
def read_synset_mapping(mapping_path):  #synset:同义词集合
    raw_bytes=mapping_path.read_bytes()
    text = raw_bytes.decode('utf-8-sig')    #将文件字节按 UTF-8 解读成文字；如果开头有 BOM 标记，就自动去掉它

    class_names={}

    for line_number,line in enumerate(text.splitlines(),start=1):
        line=line.strip()

        if not line:        #""空字符串，这个if是为了忽略空白行
            continue

        parts=line.split(maxsplit=1)

        if len(parts)!=2:
            raise ValueError(f'第{line_number}行缺少类别编号或名称')

        synset,name=parts

        if re.fullmatch(r'n[0-9]{8}',synset)is None:
            raise ValueError(f'第{line_number}行类别编号格式不正确：{synset}')

        if synset in class_names:
            raise ValueError(f'类别编号错误：{synset}')

        class_names[synset]=name

    if len(class_names)!=1000:
        raise ValueError(f'需要完整的1000类类别表，实际读取到{len(class_names)}类')

    source_sha256=hashlib.sha256(raw_bytes).hexdigest()

    return class_names,source_sha256

#固定抽取类别，建立从0开始的标签映射
def select_classes(mapping_path,output_dir,num_classes,seed):
    class_names,source_sha256=read_synset_mapping(mapping_path)

    if not 1<=num_classes<=len(class_names):
        raise ValueError('选择的类别必须在1-1000之间。')

    all_synsets=sorted(class_names)     #对字典排序实际上是对键进行排序，生成的all_synsets是一个列表，列表只含有键，没有值

    generator=random.Random(seed)

    selected_synsets=sorted(
        generator.sample(all_synsets,num_classes)   #无放回抽样
    )
    classes=[]

    for class_index,synset in enumerate(selected_synsets):
        classes.append({
            'class_index':class_index,
            'synset':synset,
            'name':class_names[synset]
        })

    #为所选类别信息生成摘要
    class_list_bytes=json.dumps(
        classes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',',':')    #指定 JSON 中使用：逗号分隔元素，冒号分隔键和值。
    ).encode('utf-8')

    class_list_sha256=hashlib.sha256(class_list_bytes).hexdigest()

    #组织最终要保存的信息
    result={
        "schema_version":1,     #schema:数据结构规则
        "dataset":"ImageNet self-selected subset",
        "num_classes":num_classes,
        "seed":seed,
        "selection_method":(
            "Sort all synsets,sample without replacement with random.Random(seed),then sort selected synsets."
        ),
        "source_file":mapping_path.name,
        "source_sha256":source_sha256,
        "class_list_sha256":class_list_sha256,
        "classes":classes
    }

    #保存映射，避免意外覆盖已经固定的类别方案
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path=output_dir/"classes.json"

    if output_path.exists():
        existing_result=json.loads(
            output_path.read_text(encoding='utf-8')
        )
        if existing_result!=result:
            raise RuntimeError('classes.json已经存在且内容不同，停止写入，避免改变固定类别')
        print('已有类别表与本次结果一致，不重复写入')

    else:
        with output_path.open('x',encoding='utf-8')as output_file:  #x:创建新文件
            json.dump(result,output_file,ensure_ascii=False,indent=2)
            output_file.write('\n')
        print('类别表已保存：',output_path)

    print('原始类别数：',len(class_names))
    print("选择类别数",len(classes))
    print('随机种子:',seed)
    print('类别列表SHA-256',class_list_sha256)
    print("前5个类别：")

    for row in classes[:5]:
        print(
            f"标签={row['class_index']},编号={row['synset']},名称={row['name']}"
        )

#保存审计结果
def save_audit(path,result):
    path.write_text(
        json.dumps(
        result,
        ensure_ascii=False,
        indent=2
        )+'\n',
        encoding='utf-8'
    )

#保存固定图片清单
def save_manifest(path,rows):   #manifest:清单；rows：多条记录
    fields=[
        'relative_path',
        'class_index',
        'synset',
        'sha256'
    ]

    with path.open('x',encoding='utf-8',newline='')as output_file:
        writer=csv.DictWriter(output_file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

#只读检查原图，按固定规则划分，不移动或删除图片。例如，同一张图片仍然放在原位置，只是在train.csv或val.csv中记录它属于哪个集合
def audit_and_split(
        source_root,    #原始图片根目录
        classes_path,   #固定类别表路径
        package_info_path,  #数据包JSON信息路径
        output_dir,     #审计与划分结果保存目录
        val_ratio,     #验证集比例
        seed
):
    if not 0<val_ratio<1:
        raise ValueError('验证集比例必须在0和1之间')
    source_root=source_root.resolve(strict=True)
    output_dir=output_dir.resolve()

    if output_dir.is_relative_to(source_root):
        raise ValueError('输出目录不能放在原始图片目录里面')

    #检查已经冻结的类别映射
    class_info=json.loads(
        classes_path.read_text(encoding='utf-8')
    )   #classes_path相当于classes.json的路径
    classes=class_info['classes']

    indices=[row['class_index'] for row in classes]
    synsets=[row['synset'] for row in classes]

    if (
        not classes 
        or class_info['num_classes'] != len(classes)
        or indices !=list(range(len(classes)))
        or len(set(synsets))!=len(classes)
    ):
        raise ValueError('固定类别表不合法！')

    class_bytes=json.dumps(
        classes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',',':')
    ).encode('utf-8')

    if (hashlib.sha256(class_bytes).hexdigest()!=class_info['class_list_sha256']):
        raise ValueError('固定类别表的摘要不匹配')

    class_to_index = dict(zip(synsets,indices))        #类别编号到数字标签的对应关系,zip()：把两个序列的对应位置配对,dict()：把这些配对建立成字典
    '''
    synsets = ["n01498041", "n01629819"]
    indices = [0, 1]
    得到：
    {
    "n01498041": 0,
    "n01629819": 1,
    }
    '''

    #检查打包信息与原始图片清单
    package_info=json.loads(package_info_path.read_text(encoding='utf-8'))  #读取的是imagenet100_packages.json

    if (
        package_info['status']!='complete'
        or package_info['class_to_index']!=class_to_index
        or package_info['selected_synsets']!=synsets
    ):
        raise ValueError('打包信息未完成或固定类别表不同。')

    manifest_name=package_info['manifest']  #拿到文件名

    if (
        not isinstance(manifest_name,str)   #检查是否是字符串
        or not re.fullmatch(r'[A-Za-z0-9_.-]+',manifest_name)  #+表示至少一个字符，可重复
    ):
        raise ValueError('打包清单文件名不合法。')

    package_manifest=package_info_path.parent/manifest_name
    manifest_bytes=package_manifest.read_bytes()

    if (hashlib.sha256(manifest_bytes).hexdigest()!=package_info['manifest_sha256']):
        raise ValueError('打包清单摘要不匹配')

    with package_manifest.open(
        'r',    #只读
        encoding='utf-8-sig',   #csv也用
        newline=''
    )as file:
        reader=csv.DictReader(file)
        required={
            'archive_path',     #归档文件路径，这里指tar包内某张图片的位置
            'size_bytes',
            'synset',
            'class_index'
        }

        if not required.issubset(reader.fieldnames or []):  #[]是为了避免fieldnames里面没有值，可能得到None,不是一个列表，不写[]可能直接报错
            raise ValueError('打包清单缺少必要列。')

        source_rows=list(reader)    #读取csv中的所有数据行，保存为列表

    #逐条检查图片路径和标签
    expected_paths=set()
    source_counts=Counter()

    for row in source_rows:
        relative=PurePosixPath(row['archive_path'])

        if (
            relative.is_absolute()  #拒绝绝对路径，absolute意为绝对的
            or "\\"in row['archive_path']   #\\表示一个反斜杠\，拒绝混入Windows风格的路径分隔符，统一使用/
            or '..' in relative.parts
            or len(relative.parts)!=3
            or relative.parts[0] not in {'train','val'}
            or relative.parts[1]!=row['synset']
            or class_to_index.get(row['synset'])!=int(row['class_index'])
            or int(row['size_bytes'])<=0
        ):
            raise ValueError(f"打包清单路径或标签不合法：{row['archive_path']}")

        image_path=(source_root/relative).resolve()
        if not image_path.is_relative_to(source_root):
            raise ValueError('图片路径超出原始数据目录。')
        if row['archive_path'] in expected_paths:
            raise ValueError('打包清单存在重复路径。')
        expected_paths.add(row['archive_path'])
        source_counts[relative.parts[0]]+=1

    if (
        source_counts['train']!=package_info['source_train_count']
        or source_counts['val']!=package_info['source_val_as_project_test_count']):
        raise ValueError('打包信息与清单图片数不同。')

    actual_path={
        path.relative_to(source_root).as_posix()for 
        path in source_root.rglob('*')if path.is_file()
        }   
    #relative_to意思为去掉共同根目录
    #glob是按照模式匹配路径，前面的r表示递归查找
    output_dir.mkdir(parents=True,exist_ok=False)
    audit_path=output_dir/'data_audit.json'
    audit={
            'schema_version':1,
            'status':'auditing',
            'source_root':str(source_root),
            'seed':seed,
            'val_ratio':val_ratio,
            'class_list_sha256':class_info['class_list_sha256'],
            'source_manifest_sha256':package_info['manifest_sha256'],
            'source_counts':dict(source_counts),
            'processed_count':0,
            'missing_files':sorted(expected_paths-actual_path),
            'unexpected_files':sorted(actual_path-expected_paths),
            'bad_files':[]
    }
    save_audit(audit_path,audit)
    if audit['missing_files']or audit['unexpected_files']:
        audit['status']='blocked'
        save_audit(audit_path,audit)
        raise RuntimeError('图片路径不齐或存在额外文件，请查看data_audit.json')

    valid_rows=[]

    for row_number ,row in enumerate(source_rows,start=1):
        relative_path=row['archive_path']
        try:
            raw_bytes=(source_root/relative_path).read_bytes()
            if len(raw_bytes)!=int(row['size_bytes']):
                    raise ValueError('图片大小与打包清单不同')
            with Image.open(BytesIO(raw_bytes))as source_image:     #BytesIO把字节包装成一个像文件一样可读取的内存对象。
                with source_image.convert('RGB')as rgb_image:
                    rgb_image.load()
            valid_rows.append({
                'relative_path':relative_path,
                'class_index':int(row['class_index']),
                'synset':row['synset'],
                'sha256':hashlib.sha256(raw_bytes).hexdigest()

                })
        except Exception as error:
            audit['bad_files'].append({
                'relative_path':relative_path,
                'error':f"{type(error).__name__}:{error}"
                })
        audit['processed_count']=row_number
        if (row_number%1000==0 or row_number==len(source_rows)):
            print(f"图片审计={row_number}/{len(source_rows)},异常={len(audit['bad_files'])}",flush=True)
            save_audit(audit_path,audit)

    if audit['bad_files']:
        audit['status']='blocked'
        save_audit(audit_path,audit)
        raise RuntimeError('发现异常图片，未生成划分;请先检查审计记录')

    # 只筛选清单记录，不删除原图。
    content_groups = defaultdict(list)
    for row in valid_rows:
        content_groups[row["sha256"]].append(row)

    cleaned_rows = []
    excluded_rows = []

    for content_hash in sorted(content_groups):
        group = sorted(
            content_groups[content_hash],
            key=lambda row: (
                not row["relative_path"].startswith("val/"),
                row["relative_path"],
            ),
        )

        labels = {row["class_index"] for row in group}

        if len(labels) > 1:
            for row in group:
                excluded_rows.append({
                    **row,
                    "reason": "conflicting_labels",
                    "kept_relative_path": None,
                })
        else:
            kept_row = group[0]
            cleaned_rows.append(kept_row)

            for row in group[1:]:
                excluded_rows.append({
                    **row,
                    "reason": "duplicate_same_label",
                    "kept_relative_path": kept_row["relative_path"],
                })

    audit["cleaning"] = {
        "policy": "exclude_all_conflicting_labels; deduplicate_same_label; prefer_official_val_then_path",
        "before_count": len(valid_rows),
        "after_count": len(cleaned_rows),
        "excluded_count": len(excluded_rows),
        "excluded_rows": excluded_rows,
    }

    valid_rows = cleaned_rows
    save_audit(audit_path, audit)

    print(
        f"清单筛选完成：保留{len(valid_rows)}张，"
        f"排除{len(excluded_rows)}张；原图未删除。",
        flush=True,
    )


    #原始train按类别划分，原始val留作项目最终测试
    train_by_class=defaultdict(list)    #defaultdict(list) 是一种特殊字典：访问不存在的键时，自动为这个键建立一个新的空列表。
    test_rows=[]
    for row in valid_rows:
        if row['relative_path'].startswith('train/'):
            train_by_class[row['synset']].append(row)
        else:
            test_rows.append(row)

    train_rows=[]
    val_rows=[]
    per_class_counts=[]
    for class_row in classes:
        synset=class_row['synset']
        class_rows=sorted(
            train_by_class[synset],
            key=lambda row:row['relative_path']
        )
        test_count=sum(row['synset']==synset for row in test_rows)
        if len(class_rows)<2 or test_count==0:
            audit['status']='blocked'
            save_audit(audit_path,audit)
            raise ValueError(f'类别{synset}的训练来源或最终测试样本不足。')
        random.Random(seed).shuffle(class_rows)     #shuffle:打乱
        val_count=max(1,int(len(class_rows)*val_ratio))
        if val_count>=len(class_rows):
            raise ValueError('划分后某类别没有训练图片。')
        val_rows.extend(class_rows[:val_count])
        train_rows.extend(class_rows[val_count:])
        per_class_counts.append({
            'synset':synset,
            'train':len(class_rows)-val_count,
            'val':val_count,
            'test':test_count
        })

    #检查跨集合重复内容和同内容不同标签
    splits={
        'train':train_rows,
        'val':val_rows,
        'test':test_rows
    }
    hash_group=defaultdict(list)
    for split_name,rows in splits.items():
        for row in rows:
            hash_group[row['sha256']].append({
                'split':split_name,
                'relative_path':row['relative_path'],
                'class_index':row['class_index']
            })
    duplicate_groups=[
        group for group in hash_group.values() if len(group)>1
    ]   #duplicate：重复的

    blocking_groups=[
        group for group in duplicate_groups if( len({item['split']for item in group})>1 or len({item['class_index']for item in group})>1)
    ]

    audit['duplicate_groups']=duplicate_groups
    audit['blocking_duplicate_groups']=blocking_groups
    audit['per_class_counts']=per_class_counts
    audit['split_counts']={
        name :len(rows)for name,rows in splits.items()
    }
    if blocking_groups:
        audit['status']='blocked'
        save_audit(audit_path,audit)
        raise RuntimeError('发现跨集合重复图片或同内容不同标签，未生成划分')

    #全部通过后保存三分清单与摘要
    split_hashes={}
    for split_name,rows in splits.items():
        rows.sort(key=lambda row:row['relative_path'])
        manifest_path=output_dir/f'{split_name}.csv'
        save_manifest(manifest_path,rows)
        split_hashes[split_name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    audit['manifest_sha256']=split_hashes
    audit['status']='complete'
    save_audit(audit_path,audit)
    print('数据审计与固定划分完成：',output_dir)
    print("各集合图片数：",audit['split_counts'])

#读取参数，选择类别选择与图片审计划分模式
def main():
    project_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description='为独立ResNet50项目准备类别和数据清单')
    mode=parser.add_mutually_exclusive_group(required=True)        #下面两种模式必须选一种，而且只能选择一种
    mode.add_argument('--select-classes',action='store_true')#action="store_true" 表示：命令中出现该开关，结果为 True。没出现，为 False。
    mode.add_argument('--audit-and-split',action='store_true')
    parser.add_argument('--synsets-file',default='metadata/LOC_synset_mapping.txt')
    parser.add_argument('--num-classes',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',default='metadata')
    parser.add_argument('--source-root',default='data/imagenet100_raw')
    parser.add_argument('--classes-file',default='metadata/classes.json')
    parser.add_argument(
        "--packages-info",
        default=(
        "data/packages/imagenet100_packages/imagenet100_packages.json")
    )

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-output-dir",default="metadata/splits_seed42")
    args=parser.parse_args()


    if args.select_classes:
        select_classes(
            mapping_path=project_dir/args.synsets_file,
            output_dir=project_dir/args.output_dir,
            num_classes=args.num_classes, seed=args.seed
        )
    else:
        audit_and_split(
            source_root=project_dir/args.source_root,
            classes_path=project_dir/args.classes_file,
            package_info_path=project_dir/args.packages_info,
            output_dir=project_dir/args.split_output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

if __name__=='__main__':
    main()
        

    





















# #读取参数并执行类别选择
# def main():
#     project_dir=Path(__file__).resolve().parent

#     parser=argparse.ArgumentParser(
#         description="为独立ResNet50项目准备类别和数据清单"
#     )
#     parser.add_argument(
#         "--select-classes",
#         action="store_true",
#         required=True
#     )
#     parser.add_argument(
#         '--synsets-file',
#         default='metadata/LOC_synset_mapping.txt'
#     )
#     parser.add_argument('--num-classes',type=int,default=100)
#     parser.add_argument('--seed',type=int,default=42)
#     parser.add_argument('--output-dir',default="metadata")
#     args=parser.parse_args()

#     mapping_path=project_dir/args.synsets_file
#     output_dir=project_dir/args.output_dir

#     select_classes(
#         mapping_path=mapping_path,
#         output_dir=output_dir,
#         num_classes=args.num_classes,seed=args.seed
#     )

# if __name__=='__main__':
#     main()



