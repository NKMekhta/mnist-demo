FROM rust:slim AS builder
WORKDIR /app
COPY . .
RUN ./scripts/copy-model.sh
RUN cargo build --release --bin=inference-server

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/model.mpk .
COPY --from=builder /app/target/release/inference-server .
EXPOSE 8080
CMD ["./inference-server"]
