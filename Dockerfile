# Build with Bookworm's compiler so its C++ ABI matches both runtime images.
FROM debian:bookworm-slim AS compiler-build
RUN apt-get update && apt-get install -y --no-install-recommends g++ make && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY compiler/Makefile ./
COPY compiler/include ./include
COPY compiler/src ./src
RUN make

# The native compiler (compiler/mlir): nova-compile, which runs quantum programs and OpenQASM circuits through MLIR
# and LLVM with its JIT (--run). Built against LLVM/MLIR 20 from apt.llvm.org, the release the MLIR CI job uses.
FROM debian:bookworm-slim AS native-build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget gnupg g++ cmake ninja-build \
    && wget -qO /etc/apt/trusted.gpg.d/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
    && echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends llvm-20-dev libmlir-20-dev mlir-20-tools \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY compiler/include ./compiler/include
COPY compiler/src ./compiler/src
COPY compiler/mlir ./compiler/mlir
RUN cmake -G Ninja -S compiler/mlir -B build -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/bin/true \
    && ninja -C build nova-compile

FROM debian:bookworm-slim AS worker
# libllvm20 (LLVM's shared library, which nova-compile links) from the same signed repository
COPY --from=native-build /etc/apt/trusted.gpg.d/apt.llvm.org.asc /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends libstdc++6 coreutils libllvm20 \
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/llvm.list
WORKDIR /opt/nova
COPY --from=compiler-build /src/build/nova ./nova
COPY --from=native-build /src/build/bin/nova-compile ./nova-compile
COPY compiler/stdlib ./stdlib
ENV NOVA_HOME=/opt/nova
USER 65534:65534
ENTRYPOINT ["/opt/nova/nova"]

FROM node:22-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM node:22-bookworm AS backend-deps
WORKDIR /app
COPY server/package*.json ./
RUN npm ci --omit=dev

FROM docker:27-cli AS docker-client

FROM node:22-bookworm-slim AS server
ENV NODE_ENV=production NOVA_EXECUTION_MODE=docker NOVA_TEMP_DIR=/var/lib/novacomp/jobs \
    NOVA_HOME=/opt/nova NOVA_COMPILER_PATH=/opt/nova/nova NOVA_STATIC_DIR=/app/dist
WORKDIR /app
COPY --from=docker-client /usr/local/bin/docker /usr/local/bin/docker
COPY --from=backend-deps /app/node_modules ./node_modules
COPY --from=compiler-build /src/build/nova /opt/nova/nova
COPY compiler/stdlib /opt/nova/stdlib
COPY server/package*.json ./
COPY server/src ./src
COPY server/public ./public
COPY --from=frontend-build /frontend/dist ./dist
RUN mkdir -p _tmp workspace /var/lib/novacomp/jobs && chown -R node:node /app /var/lib/novacomp
USER node
EXPOSE 3001
CMD ["node", "src/server.js"]
