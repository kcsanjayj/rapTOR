.PHONY: build test evals serve screenshots
build:
	cargo build --release
test:
	cargo test && cd python && python3 -m unittest discover -s tests
evals: build
	cd python && python3 -m dragonite.evals
serve: build
	./target/release/dragonite serve
