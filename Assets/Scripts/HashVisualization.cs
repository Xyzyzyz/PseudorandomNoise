using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static Unity.Mathematics.math;

public class HashVisualization : MonoBehaviour {

	static readonly int
		hashesId = Shader.PropertyToID("_Hashes"),
		configId = Shader.PropertyToID("_Config");

	[SerializeField]
	Mesh instanceMesh;

	[SerializeField]
	Material material;

	[SerializeField, Range(0, 1000)]
	int seed = 100;

	[SerializeField, Range(1, 512)]
	int resolution = 16;

	[SerializeField, Range(-2f, 2f)]
	float verticalOffset = 1f;

	NativeArray<uint> hashes;

	ComputeBuffer hashesBuffer;

	MaterialPropertyBlock propertyBlock;

	void OnValidate () {
		if (enabled && hashesBuffer != null) {
			OnDisable();
			OnEnable();
		}
	}

	void OnDisable () {
		hashes.Dispose();
		hashesBuffer.Release();
		hashesBuffer = null;
	}

	void OnEnable () {
		int length = resolution * resolution;
		hashes = new NativeArray<uint>(length, Allocator.Persistent);
		hashesBuffer = new ComputeBuffer(length, 4); // Length and size of each element -- uints are 32 bit AKA 4 byte
		DoJob();
	}

	void DoJob () {
		new HashJob {
			hashes = hashes,
			hash = SmallXXHash.Seed(seed),
			resolution = resolution,
			invResolution = 1f / resolution
		}.ScheduleParallel(hashes.Length, resolution, default).Complete();

		hashesBuffer.SetData(hashes);

		propertyBlock ??= new MaterialPropertyBlock(); // If propertyBlock is null, initialize it as a new property block
		propertyBlock.SetBuffer(hashesId, hashesBuffer);
		propertyBlock.SetVector(configId, new Vector4(resolution, 1f / resolution, verticalOffset / resolution));
	}

	void Update () {
		Graphics.DrawMeshInstancedProcedural(
			instanceMesh, 0, material, new Bounds(Vector3.zero, Vector3.one),
			hashes.Length, propertyBlock
		);
	}

	[BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
	struct HashJob : IJobFor {
		
		[WriteOnly]
		public NativeArray<uint> hashes;

		public SmallXXHash hash;

		public int resolution;
		public float invResolution;

		public void Execute (int i) {
			int v = (int)floor(invResolution * i + 0.00001f); // Same as in HashGPU.hlsl
			int u = i - resolution * v; // Same as in HashGPU.hlsl

			v -= resolution / 2;
			u -= resolution / 2;
			
			hashes[i] = hash.Eat(u).Eat(v);
		}
	}

	
}