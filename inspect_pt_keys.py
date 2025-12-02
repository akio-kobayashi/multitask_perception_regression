import torch
import argparse
import os

def inspect_pt_file(file_path: str):
    """
    .ptファイルを読み込み、それが辞書形式であればキーの一覧を表示します。
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return

    try:
        loaded_data = torch.load(file_path, map_location='cpu')

        if isinstance(loaded_data, dict):
            print(f"ファイル '{file_path}' は辞書形式です。キー一覧:")
            for key in loaded_data.keys():
                print(f"- '{key}'")
                # もしキーがテンソルであれば、その形状も表示
                if isinstance(loaded_data[key], torch.Tensor):
                    print(f"  形状: {loaded_data[key].shape}")
            # 特定のキーが存在するか確認
            if 'hubert_feats' in loaded_data:
                print(f"  'hubert_feats' キーが存在します。形状: {loaded_data['hubert_feats'].shape}")
            if 'hubert' in loaded_data:
                print(f"  'hubert' キーが存在します。形状: {loaded_data['hubert'].shape}")
        elif isinstance(loaded_data, torch.Tensor):
            print(f"ファイル '{file_path}' は直接テンソルが保存されています。形状: {loaded_data.shape}")
        else:
            # このelseブロックは、isinstance(loaded_data, dict)がFalseの場合にのみ実行されるはず
            print(f"ファイル '{file_path}' は辞書でもテンソルでもない形式です。型: {type(loaded_data)}")

    except Exception as e:
        print(f"エラー: ファイル '{file_path}' の読み込み中に問題が発生しました: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HuBERT特徴量ファイル (.pt) のキー一覧を表示します。")
    parser.add_argument('pt_file', type=str, help="検査する.ptファイルのパス")
    args = parser.parse_args()

    inspect_pt_file(args.pt_file)
