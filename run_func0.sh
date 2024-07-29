#!/bin/bash

if [ -f "infer4noanswer_1.py" ]; then
	python infer4noanswer_1.py --func ADD
	python infer4noanswer_1.py --func RECONFIRM
	python infer4noanswer_1.py --func EX_V_CUR
	python infer4noanswer_1.py --func ASSUMPTION
	python infer4noanswer_1.py --func ALL
	python infer4noanswer_1.py --func PATCH
fi
