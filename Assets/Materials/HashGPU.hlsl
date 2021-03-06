#if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
	StructuredBuffer<uint> _Hashes;
#endif

float4 _Config;

void ConfigureProcedural () {
	#if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
		float v = floor(unity_InstanceID * _Config.y + 0.00001); // Row index // Add slight positive bias to account for floating-point errors that leave us with slightly less than the numbers we want sometimes
		float u = unity_InstanceID - (v * _Config.x);  			 // Column index

		unity_ObjectToWorld = 0.0;
		unity_ObjectToWorld._m03_m13_m23_m33 = float4(
			(u + 0.5) * _Config.y - 0.5,
			_Config.z * ((1.0 / 255.0) * (_Hashes[unity_InstanceID] >> 24) - 0.5),
			(v + 0.5) * _Config.y - 0.5,
			1.0
		);
		unity_ObjectToWorld._m00_m11_m22 = _Config.y;
	#endif
}

float3 GetHashColor () {
	#if defined(UNITY_PROCEDURAL_INSTANCING_ENABLED)
		uint hash = _Hashes[unity_InstanceID];
		return (1.0 / 255.0) * float3(
			hash & 255,
			(hash >> 8) & 255,
			(hash >> 16) & 255
		);
		//return _Config.y * _Config.y * hash;
	#else
		return 1.0;
	#endif
}

void ShaderGraphFunction_float (float3 In, out float3 Out, out float3 Color) {
	Out = In;
	Color = GetHashColor();
}

void ShaderGraphFunction_half (half3 In, out half3 Out, out half3 Color) {
	Out = In;
	Color = GetHashColor();
}