# Mummy_device
## 基本コンセプト
お母さんの見守りデバイス．子供が首から提げて使う．

子供の視界に車などの危険な物が映った時に，お母さんにアラームを送る．


## 技術
参照サイトはここ．https://qiita.com/yyoshiaki/items/058250321dc8ac1ddea6

tensorflow の objectdetection プログラムを利用している．
object_detectionフォルダは，ほぼ，`/tensorflow/models/research/object_detection` のコピーである．

なお，実行ファイルは `mummy.py`であるが，これは`object_detection_tutorial,ipynb`を参考に作成されている．

主な変更点は，モデルダウンロード作業の省略と，モデル読み込み時間の短縮のため，変数`od_graph_def`をpickleにより外部ファイル化した（`od_graph_def.pkl`）．