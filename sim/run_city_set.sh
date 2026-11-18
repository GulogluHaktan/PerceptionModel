#!/bin/bash
# Renders the base city set: 3 lighting presets x trajectories (drone / rover / inverted camera), 40 frames each, same city per seed.
cd /home/haktan/Projects/PerceptionModel
i=0
for spec in "noon drone" "noon rover" "overcast drone" "overcast inverted" "dusk drone" "dusk rover"; do
  set -- $spec; i=$((i+1)); name="city_$1_$2"
  echo "{\"name\":\"$name\",\"light\":\"$1\",\"traj\":\"$2\",\"frames\":40,\"seed\":7}" > runs/city_job/config.json
  bash sim/run_isaac.sh sim/isaac_city.py > runs/city_$name.log 2>&1; echo "$name EXIT $?"
done
echo ALLDONE
