In order to support distributed training, four options are added

--jobType  either 'ps' for parameter server or 'worker' for worker
--psHosts  list of ps hosts
--workerHosts list of worker hosts
--taskId  the index of this tf process in either ps or worker list

Example,

* Running parameter server, on machine 135.104.238.72

flow  --jobType 'ps' \
  --psHosts '135.104.238.72:2222' \
  --taskId 0 \
  --workerHosts '135.104.238.169:2222,135.104.238.165:2222'

* Running worker 0 on machine 135.104.238.169, this is the chief worker

flow --model cfg/yolo-voc.cfg --train --dataset "./VOCdevkit/VOC2007/JPEGImages" --annotation ./VOCdevkit/VOC2007/Annotations/ \
 --jobType 'worker' \
  --psHosts '135.104.238.72:2222' \
  --taskId 0 \
  --workerHosts '135.104.238.169:2222,135.104.238.165:2222'

* Running worker 1 ob machine 135.104.238.165

# on blipp15:2222
flow --model cfg/yolo-voc.cfg --train --dataset "./VOCdevkit/VOC2007/JPEGImages" --annotation ./VOCdevkit/VOC2007/Annotations/ \
 --jobType 'worker' \
  --psHosts '135.104.238.72:2222' \
  --taskId 1 \
  --workerHosts '135.104.238.169:2222,135.104.238.165:2222'





